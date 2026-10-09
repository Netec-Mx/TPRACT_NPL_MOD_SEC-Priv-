# Taller Práctico   Procesamiento del lenguaje natural con modelos de secuencias

# Construcción de un sistema para analizar y detectar solicitudes bancarias duplicadas

## Metadatos

| Campo | Valor |
|-------|-------|
| **Duración del laboratorio** | 105 minutos |
| **Preparación del entorno** | 30-45 minutos la primera vez |
| **Complejidad** | Alta |
| **Nivel Bloom** | Crear |
| **Máquina objetivo** | Windows Server con experiencia de escritorio, Visual Studio Code |

## Descripción General

En esta práctica integradora construirás un sistema de detección de solicitudes bancarias duplicadas. Partirás de un dataset sintético en español, explorarás los textos, normalizarás el lenguaje, entrenarás embeddings Word2Vec y ajustarás una red siamesa con un LSTM compartido. La etiqueta positiva significa que dos textos salen del mismo grupo de redacción: expresan la misma solicitud de origen con otras palabras.

El entrenamiento, la validación y la prueba usan grupos distintos. Los callbacks ven solo la validación. Las métricas finales se calculan sobre la prueba, junto a una línea base de solapamiento de tokens. Al terminar tendrás un modelo que recibe dos solicitudes y devuelve la probabilidad de que sean duplicadas.

## Objetivos de Aprendizaje

- [ ] Implementar un pipeline de preprocesamiento para textos bancarios en español (normalización, tokenización, stopwords y padding)
- [ ] Entrenar embeddings Word2Vec solo con los grupos de entrenamiento y construir una matriz de pesos compatible con Keras
- [ ] Separar los grupos antes de armar pares, de modo que una misma solicitud no aparezca en entrenamiento y en evaluación
- [ ] Implementar una red siamesa con LSTM compartido y distancia L1 para clasificar pares de solicitudes
- [ ] Entrenar el modelo con un conjunto de validación distinto del de prueba, usando early stopping, checkpoint y reducción del learning rate
- [ ] Evaluar con accuracy, precisión, recall, F1-score y ROC-AUC, comparar contra una línea base léxica e interpretar el resultado en contexto bancario

## Prerrequisitos

### Conocimientos requeridos

| Tema | Nivel |
|------|-------|
| Python 3.10 y notebooks en Visual Studio Code | Intermedio |
| PowerShell | Básico |
| NumPy y pandas | Intermedio |
| Redes neuronales recurrentes (LSTM) | Conceptual sólido |
| Redes siamesas y clasificación binaria | Conceptual |
| Word embeddings (Word2Vec) | Conceptual |
| Keras / TensorFlow (ciclo train / eval) | Básico-Intermedio |

### Máquina virtual

| Componente | Especificación |
|------------|----------------|
| Sistema | Windows Server 2019 o 2022, **64 bits, con experiencia de escritorio** |
| CPU | 4 núcleos (Intel Core i5 8ª gen / AMD Ryzen 5 3000+, o vCPU equivalentes) |
| RAM | 8 GB (recomendado 16 GB) |
| Disco libre | 10 GB en el volumen del perfil de usuario |
| GPU | No requerida. TensorFlow se usa en CPU |
| Red | Salida a Internet hacia `python.org`, `code.visualstudio.com`, `aka.ms`, `pypi.org` y `files.pythonhosted.org` |
| Permisos | Usuario local con permiso de escritura en su perfil. Un administrador debe habilitar rutas largas e instalar el redistribuible de Visual C++ |

Windows Server Core, una sesión solo con PowerShell remoto, Python de 32 bits y Python 3.12 quedan fuera de este laboratorio: Visual Studio Code necesita escritorio, y TensorFlow 2.15 publica ruedas para Python 3.10 y 3.11 de 64 bits.

### Software

| Herramienta | Versión | Para qué |
|-------------|---------|----------|
| Visual Studio Code | 1.90 o posterior, instalador de sistema 64 bits | Editar y ejecutar el notebook |
| Extensión Python | `ms-python.python` | Intérprete y entorno virtual |
| Extensión Jupyter | `ms-toolsai.jupyter` | Celdas del notebook |
| Python | 3.10.11 o 3.10.12, 64 bits, con py launcher y Add to PATH | Runtime del laboratorio |
| Microsoft Visual C++ Redistributable 2015-2022 x64 | Vigente | TensorFlow carga sus DLL con este runtime |
| Microsoft Edge WebView2 Runtime | Evergreen x64 | Visual Studio Code en Windows Server |
| PowerShell | 5.1, ya incluido en Windows Server | Preparar carpetas y el entorno virtual |
| TensorFlow | 2.15.0 | Keras va incluido en esta versión. No instales el paquete `keras` suelto |
| Gast | 0.5.4 | Versión alineada con TensorFlow 2.15. Sin este pin, pip instala Gast 0.6 o posterior |
| NumPy | 1.26.4 | Tensores y la matriz de embeddings |
| pandas | 2.2.1 | Dataset tabular |
| scikit-learn | 1.4.1.post1 | Partición por grupos y métricas |
| NLTK | 3.8.1 | Stopwords y tokenización |
| Gensim | 4.3.2 | Word2Vec |
| SciPy | 1.12.0 | Álgebra que Gensim 4.3.2 importa. SciPy 1.13+ quita `scipy.linalg.triu` y rompe el arranque |
| Matplotlib / seaborn | 3.8.3 / 0.13.2 | Gráficos |
| tqdm | 4.66.2 | Barra de progreso |
| Unidecode | 1.3.8 | Quitar acentos |
| ipykernel | 6.29.4 | Kernel del notebook dentro de Visual Studio Code |

## Entorno del Laboratorio

Abre **PowerShell** después de cada instalador. El laboratorio vive en el perfil del usuario de Windows que abre Visual Studio Code:

`C:\Users\<tu_usuario>\nlp_banking_lab`

Ese usuario y el usuario que cree el entorno virtual tienen que ser el mismo. Evita "Ejecutar como administrador" al abrir Visual Studio Code si el entorno se creó con tu usuario normal.

### 1. Habilitar rutas largas

TensorFlow anida archivos por debajo de los 260 caracteres. En PowerShell **ejecutado como administrador**:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Cierra sesión y vuelve a entrar antes de instalar las librerías. Si una política de dominio bloquea esta clave, pide que habiliten *Enable Win32 long paths* en la directiva del equipo.

### 2. Instalar el runtime de Visual C++ y WebView2

1. Instala [Microsoft Visual C++ Redistributable 2015-2022 x64](https://aka.ms/vs/17/release/vc_redist.x64.exe).
2. Instala [WebView2 Runtime Evergreen x64](https://developer.microsoft.com/microsoft-edge/webview2/) si Visual Studio Code no abre la ventana.

### 3. Instalar Python 3.10 de 64 bits

1. Descarga el instalador **Windows installer (64-bit)** de [Python 3.10.11](https://www.python.org/downloads/release/python-31011/) o [Python 3.10.12](https://www.python.org/downloads/release/python-31012/).
2. Marca **Add python.exe to PATH**.
3. Marca **Install launcher for all users (py)**.
4. Completa la instalación para todos los usuarios si la VM lo permite.

Cierra PowerShell, ábrelo de nuevo y comprueba:

```powershell
py -3.10 --version
py -3.10 -c "import struct, sys; bits = struct.calcsize('P') * 8; print(bits); assert bits == 64; assert sys.version_info[:2] == (3, 10)"
```

La salida de la versión debe empezar por `Python 3.10` y el segundo comando debe imprimir `64`.

### 4. Instalar Visual Studio Code y las extensiones

1. Instala [Visual Studio Code para Windows x64](https://code.visualstudio.com/download) con el **System Installer**. Marca **Add to PATH**.
2. Abre un PowerShell nuevo:

```powershell
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

Si `code` no se reconoce, abre Visual Studio Code, pulsa `Ctrl+Shift+P` y ejecuta **Shell Command: Install 'code' command in PATH**. Después abre otra terminal.

### 5. Crear carpetas, entorno virtual y librerías

En PowerShell, con tu usuario normal:

```powershell
$Base = Join-Path $env:USERPROFILE "nlp_banking_lab"
$dirs = @(
  "data\raw",
  "data\processed",
  "models",
  "notebooks",
  "embeddings",
  "nltk_data"
)
foreach ($dir in $dirs) {
  New-Item -ItemType Directory -Force -Path (Join-Path $Base $dir) | Out-Null
}
Set-Location $Base

$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$requirements = @(
  "tensorflow==2.15.0",
  "gast==0.5.4",
  "numpy==1.26.4",
  "pandas==2.2.1",
  "scikit-learn==1.4.1.post1",
  "nltk==3.8.1",
  "gensim==4.3.2",
  "scipy==1.12.0",
  "matplotlib==3.8.3",
  "seaborn==0.13.2",
  "tqdm==4.66.2",
  "Unidecode==1.3.8",
  "ipykernel==6.29.4"
)
[System.IO.File]::WriteAllLines((Join-Path $Base "requirements.txt"), $requirements, $utf8NoBom)

py -3.10 -m venv nlp_env
& .\nlp_env\Scripts\python.exe -m pip install --upgrade "pip==24.0"
& .\nlp_env\Scripts\python.exe -m pip install -r requirements.txt
& .\nlp_env\Scripts\python.exe -m ipykernel install --user --name nlp_banking_lab --display-name "Python (nlp_banking_lab)"
```

`WriteAllLines` con `UTF8Encoding $false` evita que PowerShell 5.1 guarde `requirements.txt` con BOM. Un BOM hace que pip no reconozca la primera librería.

La instalación de TensorFlow puede tardar varios minutos. El comando usa `nlp_env\Scripts\python.exe`, así que no depende de activar el entorno. Si quieres el prefijo `(nlp_env)` en la terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\nlp_env\Scripts\Activate.ps1
```

`-Scope Process` solo afecta esa ventana y no pide cambiar la política del servidor.

### 6. Abrir el notebook

1. En la carpeta del laboratorio crea el archivo `notebooks\siamese_banking_duplicates.ipynb` si aún no existe. Puedes crearlo vacío con:

```powershell
$code = '{"cells":[],"metadata":{"kernelspec":{"display_name":"Python (nlp_banking_lab)","language":"python","name":"nlp_banking_lab"}},"nbformat":4,"nbformat_minor":5}'
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText((Join-Path $Base "notebooks\siamese_banking_duplicates.ipynb"), $code, $utf8NoBom)
code (Join-Path $Base "notebooks\siamese_banking_duplicates.ipynb")
```

2. Cuando Visual Studio Code pregunte, confía en la carpeta.
3. Elige el kernel **Python (nlp_banking_lab)**. Si no aparece: `Ctrl+Shift+P` → **Python: Select Interpreter** → `nlp_banking_lab\nlp_env\Scripts\python.exe`, y vuelve a elegir el kernel del notebook.
4. Copia cada bloque Python de esta guía en una celda nueva, en orden, y ejecútala con `Shift+Enter`.

Las rutas del código son absolutas. El notebook puede quedar dentro de `notebooks` sin que eso mueva los datos ni los modelos.

### Problemas frecuentes en la VM

| Síntoma | Qué hacer |
|---------|-----------|
| `py` no se reconoce | Vuelve a instalar Python con **py launcher** y abre una terminal nueva |
| `python` abre Microsoft Store | Usa `py -3.10` o `.\nlp_env\Scripts\python.exe` |
| La activación del venv dice que la ejecución de scripts está deshabilitada | Usa el `python.exe` del entorno, o `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `DLL load failed while importing _pywrap_tensorflow_internal` | Instala el redistribuible de Visual C++ x64 y reinicia la sesión |
| `FileNotFoundError` al instalar TensorFlow | Confirma que LongPathsEnabled quedó en 1 y que reiniciaste sesión |
| pip no llega a PyPI por un proxy | `.\nlp_env\Scripts\python.exe -m pip install --proxy http://host:puerto -r requirements.txt` |
| El certificado TLS del proxy intercepta pip | Repite el install agregando `--trusted-host pypi.org --trusted-host files.pythonhosted.org` |
| Visual Studio Code abre en blanco | Instala WebView2 Runtime x64 |
| El kernel no aparece | Repite `python.exe -m ipykernel install --user ...` con el Python del venv y recarga la ventana |

---

## Verificación del entorno

### Objetivo

Confirmar que el kernel del notebook es el Python de 64 bits del entorno, con TensorFlow 2.15, antes de generar datos.

### Instrucciones

Esta es la primera celda del notebook. La variable de entorno de TensorFlow tiene que asignarse antes de importar la librería.

```python
import os
import struct
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONIOENCODING"] = "utf-8"

assert struct.calcsize("P") * 8 == 64, "Instala Python 3.10 de 64 bits"
assert sys.version_info[:2] in {(3, 10), (3, 11)}, sys.version

import gensim
import matplotlib
import nltk
import numpy as np
import pandas as pd
import seaborn
import sklearn
import tensorflow as tf
from unidecode import unidecode

tf.get_logger().setLevel("ERROR")

print(f"Python {sys.version.split()[0]} ({struct.calcsize('P') * 8} bits)")
print(f"Ejecutable: {sys.executable}")
print(f"TensorFlow {tf.__version__}")
print(f"CPU: {tf.config.list_physical_devices('CPU')}")

assert tf.__version__.startswith("2.15."), tf.__version__
assert "nlp_env" in sys.executable.replace("/", "\\").lower()
print("✓ Entorno listo para el laboratorio")
```

### Salida esperada

```text
Python 3.10.x (64 bits)
Ejecutable: C:\Users\<tu_usuario>\nlp_banking_lab\nlp_env\Scripts\python.exe
TensorFlow 2.15.0
✓ Entorno listo para el laboratorio
```

### Verificación

La celda termina en `✓ Entorno listo para el laboratorio`. Si una aserción falla, corrige el intérprete del kernel antes de seguir. Al importar TensorFlow puede aparecer una sola vez un aviso de deprecación sobre `sparse_softmax_cross_entropy`; el laboratorio sigue con normalidad. `tf.get_logger().setLevel("ERROR")` oculta el resto de avisos internos de TensorFlow.

---

## Paso 1: Generación del dataset sintético de solicitudes bancarias

### Objetivo

Crear solicitudes bancarias en español agrupadas por duplicado. Cada grupo contiene una solicitud base y redacciones distintas de esa misma solicitud.

### Instrucciones

1. Prepara semillas, rutas y carpetas:

```python
import json
import random
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.join(os.path.expanduser("~"), "nlp_banking_lab")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
NLTK_DIR = os.path.join(BASE_DIR, "nltk_data")

for ruta in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, EMBEDDINGS_DIR, NOTEBOOKS_DIR, NLTK_DIR):
    os.makedirs(ruta, exist_ok=True)

print(f"Directorio base: {BASE_DIR}")
```

2. Genera el dataset. Cada variación cambia el texto: un reemplazo que sí exista en la frase y un prefijo, un sufijo, o ambos.

```python
templates = {
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

PREFIJOS = ("Buenos días, ", "Hola, ", "Urgente: ")
SUFIJOS = (
    " por favor",
    ". Agradezco su pronta respuesta",
    ". Quedo atento",
    ". Gracias",
)
REEMPLAZOS = (
    ("Solicito", "Pido"),
    ("Necesito", "Requiero"),
    ("Quiero", "Deseo"),
    ("Deseo", "Quisiera"),
    ("Presento", "Interpongo"),
    ("Requiero", "Preciso"),
    ("información", "info"),
    ("Información", "Info"),
)


def variar_solicitud(texto):
    """Devuelve otra redacción de la misma solicitud."""
    texto_var = texto
    aplicables = [(origen, destino) for origen, destino in REEMPLAZOS if origen in texto_var]
    if aplicables:
        origen, destino = random.choice(aplicables)
        texto_var = texto_var.replace(origen, destino, 1)
    modo = random.choice(("prefijo", "sufijo", "ambos"))
    if modo in ("prefijo", "ambos"):
        texto_var = random.choice(PREFIJOS) + texto_var[0].lower() + texto_var[1:]
    if modo in ("sufijo", "ambos"):
        texto_var = texto_var + random.choice(SUFIJOS)
    if texto_var == texto:
        texto_var = texto + ". Gracias"
    return texto_var


records = []
id_counter = 0
grupo_counter = 0

for categoria, textos_base in templates.items():
    for texto_base in textos_base:
        grupo_counter += 1
        id_counter += 1
        records.append({
            "id": id_counter,
            "texto_solicitud": texto_base,
            "categoria": categoria,
            "grupo_duplicado": grupo_counter,
        })
        vistos = {texto_base}
        for _ in range(random.randint(2, 4)):
            texto_variado = variar_solicitud(texto_base)
            while texto_variado in vistos:
                texto_variado += " gracias"
            vistos.add(texto_variado)
            id_counter += 1
            records.append({
                "id": id_counter,
                "texto_solicitud": texto_variado,
                "categoria": categoria,
                "grupo_duplicado": grupo_counter,
            })

df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

output_path = os.path.join(RAW_DIR, "banking_requests.csv")
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Dataset generado: {df.shape[0]} solicitudes, {df['grupo_duplicado'].nunique()} grupos")
print(f"Guardado en: {output_path}")
df.head(10)
```

`utf-8-sig` permite que Excel, en Windows, abra los acentos sin cambiar la codificación a mano.

### Salida esperada

```text
Dataset generado: 201 solicitudes, 50 grupos
Guardado en: C:\Users\<tu_usuario>\nlp_banking_lab\data\raw\banking_requests.csv
```

Con `SEED = 42` el conteo es exactamente 201. La tabla muestra `id`, `texto_solicitud`, `categoria` y `grupo_duplicado`.

### Verificación

```python
assert df.shape[0] >= 150, "El dataset debe tener al menos 150 registros"
assert df["id"].is_unique
assert set(df.columns) == {"id", "texto_solicitud", "categoria", "grupo_duplicado"}
assert df["categoria"].nunique() == 5
assert df["grupo_duplicado"].nunique() == 50
assert df["texto_solicitud"].is_unique
print("✓ Dataset generado correctamente")
```

---

## Paso 2: Análisis exploratorio de datos (EDA)

### Objetivo

Ver cómo se reparten las categorías, el tamaño de los grupos y la longitud de los textos. La longitud que usará el modelo se fija más adelante, solo con los grupos de entrenamiento.

### Instrucciones

1. Distribución de categorías y de tamaños de grupo:

```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cat_counts = df["categoria"].value_counts()
sns.barplot(
    x=cat_counts.index,
    y=cat_counts.values,
    ax=axes[0],
    hue=cat_counts.index,
    palette="viridis",
    legend=False,
)
axes[0].set_title("Distribución de categorías de solicitudes")
axes[0].set_xlabel("Categoría")
axes[0].set_ylabel("Cantidad")
axes[0].tick_params(axis="x", rotation=45)

grupo_sizes = df.groupby("grupo_duplicado").size()
sns.histplot(grupo_sizes, bins=10, ax=axes[1], color="steelblue")
axes[1].set_title("Tamaño de grupos de duplicados")
axes[1].set_xlabel("Solicitudes por grupo")
axes[1].set_ylabel("Frecuencia")

plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS_DIR, "eda_distribucion.png"), dpi=100)
plt.show()

print("Estadísticas de grupos de duplicados:")
print(f"  Tamaño mínimo: {grupo_sizes.min()}")
print(f"  Tamaño máximo: {grupo_sizes.max()}")
print(f"  Tamaño promedio: {grupo_sizes.mean():.1f}")
```

2. Longitud de los textos originales, antes de quitar stopwords:

```python
df["longitud_chars"] = df["texto_solicitud"].str.len()
df["longitud_palabras"] = df["texto_solicitud"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df["longitud_palabras"], bins=20, ax=axes[0], color="coral")
axes[0].set_title("Distribución de longitud (palabras)")
axes[0].set_xlabel("Número de palabras")
axes[0].set_ylabel("Frecuencia")

sns.boxplot(
    data=df,
    x="categoria",
    y="longitud_palabras",
    ax=axes[1],
    hue="categoria",
    palette="Set2",
    legend=False,
)
axes[1].set_title("Longitud por categoría")
axes[1].set_xlabel("Categoría")
axes[1].set_ylabel("Palabras")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS_DIR, "eda_longitudes.png"), dpi=100)
plt.show()

print("Estadísticas de longitud (palabras, texto original):")
print(f"  Mínimo: {df['longitud_palabras'].min()}")
print(f"  Máximo: {df['longitud_palabras'].max()}")
print(f"  Promedio: {df['longitud_palabras'].mean():.1f}")
print(f"  Percentil 95: {df['longitud_palabras'].quantile(0.95):.0f}")
```

### Salida esperada

Dos figuras guardadas en `notebooks`. Con `SEED = 42` cada grupo tiene entre 3 y 5 solicitudes (promedio 4.0) y el texto original promedia 10.0 palabras (percentil 95: 14).

### Verificación

```python
assert "longitud_palabras" in df.columns
assert grupo_sizes.min() >= 3
assert grupo_sizes.max() <= 5
print("✓ EDA completado correctamente")
```

---

## Paso 3: Preprocesamiento de texto

### Objetivo

Normalizar las solicitudes y convertirlas en tokens. Los números se conservan: un monto, un plazo o una referencia forman parte del sentido de una solicitud bancaria. Los acentos se eliminan para que `crédito` y `credito` coincidan en el vocabulario de este laboratorio.

### Instrucciones

1. Descarga los recursos de NLTK dentro de la carpeta del laboratorio:

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.data.path.append(NLTK_DIR)
nltk.download("punkt", download_dir=NLTK_DIR, quiet=True)
nltk.download("stopwords", download_dir=NLTK_DIR, quiet=True)

stop_words_es = set(stopwords.words("spanish"))
palabras_relevantes = {"no", "sin", "contra", "sobre", "mi", "mis"}
stop_words_es = stop_words_es - palabras_relevantes

print(f"Stopwords cargadas: {len(stop_words_es)} palabras")
```

2. Define el pipeline. Después de la limpieza, `word_tokenize` y un `split` producen el mismo corte; se usa NLTK para dejar explícita la tokenización.

```python
def preprocesar_texto(texto):
    """
    Normaliza una solicitud bancaria en español.

    1. Minúsculas
    2. Sin acentos
    3. Conserva letras y números; el resto pasa a espacio
    4. Tokeniza
    5. Quita stopwords y tokens de un solo carácter
    """
    if not isinstance(texto, str):
        return []

    texto = unidecode(texto.lower())
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    tokens = word_tokenize(texto, language="spanish")
    return [t for t in tokens if t not in stop_words_es and len(t) >= 2 and t.isalnum()]


ejemplo = "Solicito información sobre créditos de libre inversión disponibles"
resultado = preprocesar_texto(ejemplo)
print(f"Original: {ejemplo}")
print(f"Procesado: {resultado}")
```

3. Aplica el pipeline y guarda el CSV. En disco, `tokens` queda como texto separado por espacios. En memoria sigue siendo una lista, que es lo que Word2Vec necesita.

```python
df["tokens"] = [
    preprocesar_texto(texto)
    for texto in tqdm(df["texto_solicitud"], desc="Preprocesando textos")
]
df["texto_limpio"] = df["tokens"].apply(lambda tokens: " ".join(tokens))
df["n_tokens"] = df["tokens"].apply(len)

print("Estadísticas post-preprocesamiento:")
print(f"  Tokens promedio: {df['n_tokens'].mean():.1f}")
print(f"  Tokens mínimo: {df['n_tokens'].min()}")
print(f"  Tokens máximo: {df['n_tokens'].max()}")

vocabulario_global = sorted({token for tokens in df["tokens"] for token in tokens})
print(f"  Vocabulario global (todas las filas, solo informativo): {len(vocabulario_global)}")

df_export = df.copy()
df_export["tokens"] = df_export["texto_limpio"]
processed_path = os.path.join(PROCESSED_DIR, "banking_requests_processed.csv")
df_export.to_csv(processed_path, index=False, encoding="utf-8-sig")
print(f"Dataset procesado guardado en: {processed_path}")

df[["texto_solicitud", "texto_limpio", "n_tokens"]].head(5)
```

### Salida esperada

```text
Stopwords cargadas: 307 palabras
Original: Solicito información sobre créditos de libre inversión disponibles
Procesado: ['solicito', 'informacion', 'sobre', 'creditos', 'libre', 'inversion', 'disponibles']

Tokens promedio: 7.4
Tokens mínimo: 4
Tokens máximo: 12
Vocabulario global (todas las filas, solo informativo): 144
```

### Verificación

```python
assert "tokens" in df.columns
assert df["tokens"].map(lambda tokens: isinstance(tokens, list)).all()
assert df["n_tokens"].min() >= 2, "Todo texto debe conservar al menos 2 tokens"
assert len(vocabulario_global) > 50
assert "credito" in vocabulario_global or "creditos" in vocabulario_global
print("✓ Preprocesamiento completado correctamente")
```

---

## Paso 4: División por grupos y embeddings Word2Vec

### Objetivo

Separar los 50 grupos en entrenamiento, validación y prueba antes de ajustar nada que aprenda del texto. Word2Vec y el vocabulario se construyen solo con el entrenamiento. Una palabra que aparezca por primera vez en validación o en prueba queda fuera de vocabulario.

### Instrucciones

1. Reparte los grupos, estratificando por categoría para que crédito, reclamo, saldo, cancelación y actualización queden en los tres cortes:

```python
from sklearn.model_selection import train_test_split

resumen_grupos = df.groupby("grupo_duplicado")["categoria"].first().reset_index()
grupos_train, grupos_temp, _, categorias_temp = train_test_split(
    resumen_grupos["grupo_duplicado"].to_numpy(),
    resumen_grupos["categoria"].to_numpy(),
    test_size=0.30,
    random_state=SEED,
    stratify=resumen_grupos["categoria"].to_numpy(),
)
grupos_val, grupos_test = train_test_split(
    grupos_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=categorias_temp,
)

grupos_train = {int(g) for g in grupos_train}
grupos_val = {int(g) for g in grupos_val}
grupos_test = {int(g) for g in grupos_test}

df_train = df[df["grupo_duplicado"].isin(grupos_train)].copy()
df_val = df[df["grupo_duplicado"].isin(grupos_val)].copy()
df_test = df[df["grupo_duplicado"].isin(grupos_test)].copy()

print(f"Grupos train/val/test: {len(grupos_train)}/{len(grupos_val)}/{len(grupos_test)}")
print(f"Solicitudes train/val/test: {len(df_train)}/{len(df_val)}/{len(df_test)}")
print("Categorías por corte:")
print(pd.DataFrame({
    "train": df_train["categoria"].value_counts(),
    "val": df_val["categoria"].value_counts(),
    "test": df_test["categoria"].value_counts(),
}).fillna(0).astype(int))
```

2. Fija la longitud de padding con el percentil 95 del entrenamiento y entrena Word2Vec. `workers=1` deja la semilla reproducible en Windows; con varios procesos Gensim no garantiza el mismo vector en cada corrida.

```python
from gensim.models import Word2Vec

MAX_SEQ_LEN = int(df_train["n_tokens"].quantile(0.95)) + 2
MAX_SEQ_LEN = max(MAX_SEQ_LEN, 8)
print(f"MAX_SEQ_LEN (solo entrenamiento): {MAX_SEQ_LEN}")

EMBEDDING_DIM = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 1
W2V_EPOCHS = 100

corpus = df_train["tokens"].tolist()
print(f"Documentos para Word2Vec: {len(corpus)}")

w2v_model = Word2Vec(
    sentences=corpus,
    vector_size=EMBEDDING_DIM,
    window=W2V_WINDOW,
    min_count=W2V_MIN_COUNT,
    workers=1,
    epochs=W2V_EPOCHS,
    seed=SEED,
    sg=1,
)

w2v_path = os.path.join(EMBEDDINGS_DIR, "banking_w2v.model")
w2v_model.save(w2v_path)
print(f"Modelo guardado en: {w2v_path}")
print(f"Vocabulario Word2Vec: {len(w2v_model.wv)} palabras")
```

La ventana de 5 tokens cubre casi todo el documento, porque tras el preprocesamiento las solicitudes son cortas. Los vecinos de una palabra muestran coocurrencia dentro de las plantillas de entrenamiento.

3. Revisa vecinos solo para palabras que sí están en el modelo:

```python
for word in ("credito", "cuenta", "tarjeta", "cancelar", "cobro"):
    if word in w2v_model.wv:
        print(f"\nPalabras cercanas a '{word}':")
        for sim_word, score in w2v_model.wv.most_similar(word, topn=5):
            print(f"  {sim_word}: {score:.3f}")
    else:
        print(f"\n'{word}' no está en el vocabulario de entrenamiento")
```

4. Construye el índice y la matriz. El índice 0 es padding y se queda en ceros para que `mask_zero` lo ignore. El índice 1, `<UNK>`, recibe el promedio de los vectores conocidos: es un vector genérico, fijo, porque la capa de embeddings no se va a reentrenar.

```python
vocabulario = sorted({token for tokens in df_train["tokens"] for token in tokens})
word_to_idx = {"<PAD>": 0, "<UNK>": 1}
idx_to_word = {0: "<PAD>", 1: "<UNK>"}
for idx, word in enumerate(vocabulario, start=2):
    word_to_idx[word] = idx
    idx_to_word[idx] = word

VOCAB_SIZE = len(word_to_idx)
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM), dtype="float32")
rng_emb = np.random.RandomState(SEED)
filas_conocidas = []

for word, idx in word_to_idx.items():
    if word in ("<PAD>", "<UNK>"):
        continue
    if word in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[word]
    else:
        embedding_matrix[idx] = rng_emb.normal(0, 0.1, EMBEDDING_DIM).astype("float32")
    filas_conocidas.append(idx)

embedding_matrix[word_to_idx["<UNK>"]] = embedding_matrix[filas_conocidas].mean(axis=0)
embedding_matrix = embedding_matrix.astype("float32")

np.save(os.path.join(EMBEDDINGS_DIR, "embedding_matrix.npy"), embedding_matrix)
with open(os.path.join(EMBEDDINGS_DIR, "word_to_idx.json"), "w", encoding="utf-8") as archivo:
    json.dump(word_to_idx, archivo, ensure_ascii=False)

print(f"Vocabulario con PAD y UNK: {VOCAB_SIZE}")
print(f"Forma de la matriz: {embedding_matrix.shape}")
print("✓ Matriz de embeddings y vocabulario guardados")
```

### Salida esperada

```text
Grupos train/val/test: 35/7/8
Solicitudes train/val/test: 138/31/32
MAX_SEQ_LEN (solo entrenamiento): 12
Documentos para Word2Vec: 138
Vocabulario Word2Vec: 118 palabras
Vocabulario con PAD y UNK: 120
Forma de la matriz: (120, 100)
```

Esas cifras corresponden a `SEED = 42`.

### Verificación

```python
assert grupos_train.isdisjoint(grupos_val)
assert grupos_train.isdisjoint(grupos_test)
assert grupos_val.isdisjoint(grupos_test)
assert set(df_train["id"]).isdisjoint(set(df_val["id"]))
assert set(df_train["id"]).isdisjoint(set(df_test["id"]))
assert set(df_val["id"]).isdisjoint(set(df_test["id"]))
assert MAX_SEQ_LEN >= 8
assert embedding_matrix.shape == (VOCAB_SIZE, EMBEDDING_DIM)
assert embedding_matrix.dtype == np.float32
assert float(embedding_matrix[0].sum()) == 0.0, "PAD debe ser el vector cero"
assert not np.allclose(embedding_matrix[1], 0.0), "UNK debe ser el promedio de los vectores"
assert word_to_idx["<PAD>"] == 0
assert word_to_idx["<UNK>"] == 1
print("✓ Grupos separados y embeddings construidos")
```

---

## Paso 5: Generación de pares dentro de cada corte

### Objetivo

Armar pares equilibrados sin mezclar grupos. Un par positivo son dos solicitudes del mismo `grupo_duplicado`. Un par negativo son dos solicitudes de grupos distintos del mismo corte. La etiqueta de este laboratorio es esa pertenencia: dos plantillas parecidas de grupos distintos se etiquetan como no duplicadas.

### Instrucciones

1. Genera los pares de entrenamiento, validación y prueba por separado:

```python
from itertools import combinations

from tensorflow.keras.preprocessing.sequence import pad_sequences


def generar_pares(dataframe, seed=42):
    """Pares 1:1 usando solamente los grupos presentes en dataframe."""
    rng = np.random.RandomState(seed)
    pairs_left = []
    pairs_right = []
    labels = []

    grupos = dataframe.groupby("grupo_duplicado")["texto_limpio"].apply(list)
    for _, textos in grupos.items():
        textos = list(textos)
        if len(textos) < 2:
            continue
        for texto_a, texto_b in combinations(textos, 2):
            pairs_left.append(texto_a)
            pairs_right.append(texto_b)
            labels.append(1)

    n_positivos = len(labels)
    group_ids = [int(grupo) for grupo in grupos.index]
    textos_por_grupo = {int(grupo): list(textos) for grupo, textos in grupos.items()}
    vistos = set()
    intentos = 0
    max_intentos = max(n_positivos * 50, 1)

    while sum(etiqueta == 0 for etiqueta in labels) < n_positivos and intentos < max_intentos:
        intentos += 1
        elegidos = rng.choice(np.array(group_ids), size=2, replace=False)
        grupo_a, grupo_b = int(elegidos[0]), int(elegidos[1])
        texto_a = textos_por_grupo[grupo_a][rng.randint(0, len(textos_por_grupo[grupo_a]))]
        texto_b = textos_por_grupo[grupo_b][rng.randint(0, len(textos_por_grupo[grupo_b]))]
        clave = (texto_a, texto_b)
        if clave in vistos:
            continue
        vistos.add(clave)
        pairs_left.append(texto_a)
        pairs_right.append(texto_b)
        labels.append(0)

    n_negativos = sum(etiqueta == 0 for etiqueta in labels)
    if n_negativos < n_positivos:
        raise RuntimeError("No hubo suficientes pares negativos dentro del corte")
    return pairs_left, pairs_right, np.array(labels, dtype="int32")


def texto_a_secuencia(texto, indice_palabras):
    """Convierte texto ya preprocesado. Lo desconocido cae en <UNK>."""
    tokens = texto.split() if isinstance(texto, str) else list(texto)
    desconocido = indice_palabras["<UNK>"]
    return [indice_palabras.get(token, desconocido) for token in tokens]


def secuencias_con_padding(textos, indice_palabras, max_len):
    secuencias = [texto_a_secuencia(texto, indice_palabras) for texto in textos]
    return pad_sequences(secuencias, maxlen=max_len, padding="post", truncating="post", value=0)


pares = {}
for nombre, parte in (("train", df_train), ("val", df_val), ("test", df_test)):
    izquierda, derecha, etiquetas = generar_pares(parte, seed=SEED)
    pares[nombre] = {"left": izquierda, "right": derecha, "labels": etiquetas}
    print(
        f"{nombre}: {len(etiquetas)} pares | "
        f"positivos {(etiquetas == 1).sum()} | negativos {(etiquetas == 0).sum()}"
    )

X_left_train = secuencias_con_padding(pares["train"]["left"], word_to_idx, MAX_SEQ_LEN)
X_right_train = secuencias_con_padding(pares["train"]["right"], word_to_idx, MAX_SEQ_LEN)
X_left_val = secuencias_con_padding(pares["val"]["left"], word_to_idx, MAX_SEQ_LEN)
X_right_val = secuencias_con_padding(pares["val"]["right"], word_to_idx, MAX_SEQ_LEN)
X_left_test = secuencias_con_padding(pares["test"]["left"], word_to_idx, MAX_SEQ_LEN)
X_right_test = secuencias_con_padding(pares["test"]["right"], word_to_idx, MAX_SEQ_LEN)
y_train = pares["train"]["labels"]
y_val = pares["val"]["labels"]
y_test = pares["test"]["labels"]

print(f"Forma train: {X_left_train.shape}, val: {X_left_val.shape}, test: {X_left_test.shape}")
```

2. Mide cuántos tokens de validación y de prueba caen en `<UNK>`. En entrenamiento la tasa tiene que ser cero, porque el vocabulario se armó con esos textos.

```python
def tasa_unk(matriz, indice_unk=1):
    tokens = matriz[matriz != 0]
    if len(tokens) == 0:
        return 0.0
    return float(np.mean(tokens == indice_unk))

print(f"Tasa UNK train: {tasa_unk(X_left_train):.2%}")
print(f"Tasa UNK val:   {tasa_unk(X_left_val):.2%}")
print(f"Tasa UNK test:  {tasa_unk(X_left_test):.2%}")
```

### Salida esperada

Con `SEED = 42`:

```text
train: 430 pares | positivos 215 | negativos 215
val: 110 pares | positivos 55 | negativos 55
test: 102 pares | positivos 51 | negativos 51
Forma train: (430, 12), val: (110, 12), test: (102, 12)
Tasa UNK train: 0.00%
```

La tasa UNK de validación y de prueba queda por encima de cero: esas palabras no entraron al vocabulario.

### Verificación

```python
for nombre in ("train", "val", "test"):
    etiquetas = pares[nombre]["labels"]
    assert abs(int((etiquetas == 1).sum()) - int((etiquetas == 0).sum())) == 0
    assert len(etiquetas) >= 20, nombre

assert X_left_train.shape[1] == MAX_SEQ_LEN
assert X_left_val.shape[1] == MAX_SEQ_LEN
assert X_left_test.shape[1] == MAX_SEQ_LEN
assert len(y_train) == len(X_left_train) == len(X_right_train)
assert not np.any(X_left_train == word_to_idx["<UNK>"])
assert not np.any(X_right_train == word_to_idx["<UNK>"])
print("✓ Pares generados dentro de cada corte")
```

---

## Paso 6: Red siamesa con LSTM compartido

### Objetivo

Construir una red que lea las dos solicitudes con la misma capa de embeddings y el mismo LSTM. La distancia L1 entre las dos codificaciones entra a una capa densa que devuelve la probabilidad de duplicado.

El LSTM es unidireccional. En CPU de Windows Server, `recurrent_dropout` cambia a una implementación mucho más lenta; el dropout queda en la entrada del LSTM. Los embeddings se congelan: con pocos cientos de pares, dejarlos entrenables borra el Word2Vec en las primeras épocas.

### Instrucciones

1. Define la distancia L1 como capa propia. Una función `lambda` dentro de `Lambda` no se puede volver a cargar con `safe_mode` de Keras 2.15, y el paso 8 necesita abrir el archivo del modelo.

```python
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Model

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# Distancia L1 elemento a elemento entre las dos codificaciones.
@tf.keras.utils.register_keras_serializable(package="laboratorio")
class DistanciaL1(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        return tf.abs(tf.subtract(inputs[0], inputs[1]))

    def get_config(self):
        return super().get_config()


def cargar_modelo(ruta):
    return tf.keras.models.load_model(
        ruta,
        custom_objects={"DistanciaL1": DistanciaL1},
    )


def crear_red_siamesa(vocab_size, embedding_dim, pesos_embedding, max_seq_len, lstm_units=64):
    entrada_izq = Input(shape=(max_seq_len,), name="input_left")
    entrada_der = Input(shape=(max_seq_len,), name="input_right")

    capa_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[pesos_embedding],
        trainable=False,
        mask_zero=True,
        name="shared_embedding",
    )
    lstm_compartido = LSTM(
        units=lstm_units,
        return_sequences=False,
        dropout=0.2,
        name="shared_lstm",
    )

    cod_izq = lstm_compartido(capa_embedding(entrada_izq))
    cod_der = lstm_compartido(capa_embedding(entrada_der))
    distancia = DistanciaL1(name="l1_distance")([cod_izq, cod_der])

    x = Dense(64, activation="relu", name="dense_1")(distancia)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu", name="dense_2")(x)
    x = Dropout(0.2)(x)
    salida = Dense(1, activation="sigmoid", name="output")(x)

    modelo = Model(inputs=[entrada_izq, entrada_der], outputs=salida, name="siamese_lstm")
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


model = crear_red_siamesa(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    pesos_embedding=embedding_matrix,
    max_seq_len=MAX_SEQ_LEN,
    lstm_units=64,
)
model.summary()

entrenables = int(sum(np.prod(peso.shape) for peso in model.trainable_weights))
congelados = int(sum(np.prod(peso.shape) for peso in model.non_trainable_weights))
print(f"Parámetros entrenables: {entrenables:,}")
print(f"Parámetros congelados (embeddings): {congelados:,}")
```

2. El diagrama es opcional. En Windows Server hace falta el binario de Graphviz, que no forma parte del entorno. Si falta, el entrenamiento sigue igual.

```python
try:
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(NOTEBOOKS_DIR, "siamese_architecture.png"),
        show_shapes=True,
        show_layer_names=True,
    )
    print("✓ Diagrama guardado en notebooks\\siamese_architecture.png")
except Exception as exc:
    print(f"Nota: no se generó el diagrama ({exc}). El entrenamiento no depende de Graphviz.")
```

### Salida esperada

El resumen muestra una capa `shared_embedding` no entrenable (12,000 parámetros), una capa `shared_lstm` usada por las dos entradas, `l1_distance` y las densas `64 → 32 → 1`. Los parámetros entrenables quedan en 48,513.

### Verificación

```python
formas = model.input_shape
if isinstance(formas, dict):
    formas = list(formas.values())
assert [tuple(forma) for forma in formas] == [(None, MAX_SEQ_LEN), (None, MAX_SEQ_LEN)]
assert tuple(model.output_shape) == (None, 1)
assert any(capa.name == "shared_lstm" for capa in model.layers)
assert not model.get_layer("shared_embedding").trainable
print("✓ Red siamesa construida correctamente")
```

---

## Paso 7: Entrenamiento

### Objetivo

Ajustar la red con los pares de entrenamiento. Early stopping, el checkpoint y la reducción del learning rate observan `val_loss`. El conjunto de prueba no entra a `fit`.

### Instrucciones

1. Entrena y guarda el mejor estado según la pérdida de validación:

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, "siamese_best.keras"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),
]

BATCH_SIZE = 32
EPOCHS = 40

print("Iniciando entrenamiento...")
print(f"  Épocas máximas: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Pares de entrenamiento: {len(y_train)}")
print(f"  Pares de validación: {len(y_val)}")
print(f"  Pares de prueba (reservados, no entran a fit): {len(y_test)}")

history = model.fit(
    x=[X_left_train, X_right_train],
    y=y_train,
    validation_data=([X_left_val, X_right_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

print(f"Entrenamiento finalizado en la época {len(history.history['loss'])}")
print(f"Mejor val_accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Mejor val_loss: {min(history.history['val_loss']):.4f}")
```

En CPU este paso suele tardar unos minutos, no los 40 pases completos: early stopping corta cuando la validación deja de mejorar.

2. Dibuja las curvas y guarda el modelo final:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["loss"], label="Train loss", linewidth=2)
axes[0].plot(history.history["val_loss"], label="Val loss", linewidth=2)
axes[0].set_title("Pérdida (binary crossentropy)")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Pérdida")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["accuracy"], label="Train accuracy", linewidth=2)
axes[1].plot(history.history["val_accuracy"], label="Val accuracy", linewidth=2)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS_DIR, "training_curves.png"), dpi=100)
plt.show()

final_model_path = os.path.join(MODELS_DIR, "siamese_final.keras")
model.save(final_model_path)
pd.DataFrame(history.history).to_csv(
    os.path.join(MODELS_DIR, "training_history.csv"),
    index=False,
    encoding="utf-8-sig",
)
print(f"Modelo final guardado en: {final_model_path}")
```

3. Comprueba que el archivo se puede volver a abrir en esta misma sesión:

```python
modelo_cargado = cargar_modelo(os.path.join(MODELS_DIR, "siamese_best.keras"))
_ = modelo_cargado.predict([X_left_val[:2], X_right_val[:2]], verbose=0)
del modelo_cargado
print("✓ Checkpoint leído de nuevo con DistanciaL1")
```

### Salida esperada

```text
Pares de entrenamiento: 430
Pares de validación: 110
Pares de prueba (reservados, no entran a fit): 102
Mejor val_accuracy por encima de 0.60
```

En CPU el early stopping puede cerrar el entrenamiento alrededor de la época 20, antes de las 40. La curva de validación corresponde a otros grupos, no al corte de prueba. Un aviso de deprecación de TensorFlow al importar no detiene este paso.

### Verificación

```python
assert len(history.history["loss"]) > 5
assert max(history.history["val_accuracy"]) > 0.60
assert os.path.exists(os.path.join(MODELS_DIR, "siamese_best.keras"))
assert os.path.exists(os.path.join(MODELS_DIR, "siamese_final.keras"))
print("✓ Entrenamiento completado correctamente")
```

---

## Paso 8: Evaluación en grupos no vistos

### Objetivo

Medir el checkpoint en el conjunto de prueba y comparar esas cifras con una línea base que solo mira el solapamiento de tokens. La línea base responde cuánto de la tarea se resuelve por palabras compartidas, sin LSTM.

### Instrucciones

1. Calcula la línea base de Jaccard y las predicciones del modelo:

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def jaccard_tokens(texto_a, texto_b):
    bolsa_a = set(texto_a.split())
    bolsa_b = set(texto_b.split())
    union = bolsa_a | bolsa_b
    if not union:
        return 0.0
    return len(bolsa_a & bolsa_b) / len(union)


UMBRAL_JACCARD = 0.5
y_base = np.array([
    1 if jaccard_tokens(texto_a, texto_b) >= UMBRAL_JACCARD else 0
    for texto_a, texto_b in zip(pares["test"]["left"], pares["test"]["right"])
], dtype="int32")
baseline_accuracy = accuracy_score(y_test, y_base)
baseline_f1 = f1_score(y_test, y_base)

best_model = cargar_modelo(os.path.join(MODELS_DIR, "siamese_best.keras"))
y_pred_proba = best_model.predict([X_left_test, X_right_test], verbose=0).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("=" * 60)
print("LÍNEA BASE LÉXICA (Jaccard >= 0.5) EN PRUEBA")
print("=" * 60)
print(f"  Accuracy: {baseline_accuracy:.4f}")
print(f"  F1-score: {baseline_f1:.4f}")

print("\n" + "=" * 60)
print("RED SIAMESA EN GRUPOS DE PRUEBA")
print("=" * 60)
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")
print("\n" + classification_report(
    y_test,
    y_pred,
    target_names=["No duplicado", "Duplicado"],
    digits=4,
    zero_division=0,
))
```

2. Matriz de confusión y curva ROC:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No duplicado", "Duplicado"],
    yticklabels=["No duplicado", "Duplicado"],
    ax=axes[0],
)
axes[0].set_title("Matriz de confusión (prueba)")
axes[0].set_xlabel("Predicción")
axes[0].set_ylabel("Real")

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
axes[1].plot([0, 1], [0, 1], "r--", linewidth=1, label="Azar")
axes[1].set_title("Curva ROC (prueba)")
axes[1].set_xlabel("Tasa de falsos positivos")
axes[1].set_ylabel("Tasa de verdaderos positivos")
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS_DIR, "evaluation_metrics.png"), dpi=100)
plt.show()
```

3. Revisa errores reconstruyendo el texto desde los índices. `<UNK>` aparece cuando la palabra no estaba en el entrenamiento.

```python
def secuencia_a_texto(secuencia):
    return " ".join(idx_to_word.get(int(indice), "?") for indice in secuencia if int(indice) != 0)

print("Falsos negativos (duplicados no detectados):", int(((y_test == 1) & (y_pred == 0)).sum()))
for i in np.where((y_test == 1) & (y_pred == 0))[0][:3]:
    print(f"  prob={y_pred_proba[i]:.3f}")
    print(f"    Izq: {secuencia_a_texto(X_left_test[i])}")
    print(f"    Der: {secuencia_a_texto(X_right_test[i])}")

print("Falsos positivos (no duplicados marcados como duplicados):", int(((y_test == 0) & (y_pred == 1)).sum()))
for i in np.where((y_test == 0) & (y_pred == 1))[0][:3]:
    print(f"  prob={y_pred_proba[i]:.3f}")
    print(f"    Izq: {secuencia_a_texto(X_left_test[i])}")
    print(f"    Der: {secuencia_a_texto(X_right_test[i])}")
```

4. Prueba redacciones nuevas. Las cuatro primeras respetan la etiqueta del laboratorio (misma solicitud de origen, o solicitudes de distinto tipo). La quinta es un caso límite: un analista podría ver dos peticiones de crédito, pero el generador las creó en grupos distintos y el modelo aprendió esa definición.

```python
def predecir_duplicado(texto1, texto2, modelo, indice_palabras, max_len):
    seq1 = texto_a_secuencia(" ".join(preprocesar_texto(texto1)), indice_palabras)
    seq2 = texto_a_secuencia(" ".join(preprocesar_texto(texto2)), indice_palabras)
    seq1 = pad_sequences([seq1], maxlen=max_len, padding="post", truncating="post", value=0)
    seq2 = pad_sequences([seq2], maxlen=max_len, padding="post", truncating="post", value=0)
    return float(modelo.predict([seq1, seq2], verbose=0)[0][0])


casos = [
    (
        "Deseo cancelar mi tarjeta de crédito por no uso",
        "Quisiera cancelar mi tarjeta de crédito por no uso, por favor",
        "DUPLICADO",
    ),
    (
        "Necesito conocer el saldo actual de mi cuenta de ahorros",
        "Requiero conocer el saldo actual de mi cuenta de ahorros. Quedo atento",
        "DUPLICADO",
    ),
    (
        "Deseo cancelar mi tarjeta de crédito por no uso",
        "Necesito financiamiento para compra de vivienda",
        "NO DUPLICADO",
    ),
    (
        "Necesito conocer el saldo actual de mi cuenta de ahorros",
        "Solicito la cancelación de mi cuenta de ahorros",
        "NO DUPLICADO",
    ),
]

print("Casos alineados con la etiqueta del laboratorio:")
for texto1, texto2, esperado in casos:
    prob = predecir_duplicado(texto1, texto2, best_model, word_to_idx, MAX_SEQ_LEN)
    resultado = "DUPLICADO" if prob >= 0.5 else "NO DUPLICADO"
    marca = "✓" if resultado == esperado else "✗"
    print(f"\n{marca} Prob: {prob:.3f} → {resultado} (esperado: {esperado})")
    print(f"  T1: {texto1}")
    print(f"  T2: {texto2}")

prob_limite = predecir_duplicado(
    "Necesito un préstamo personal para libre inversión",
    "Me interesa solicitar un crédito de consumo",
    best_model,
    word_to_idx,
    MAX_SEQ_LEN,
)
print(f"\nCaso límite entre plantillas de crédito. Prob: {prob_limite:.3f}")
print("Esas dos frases pertenecen a grupos distintos en el generador.")
print("La etiqueta con la que aprendió el modelo es no duplicado, aunque ambas pidan un crédito.")
```

### Salida esperada

Accuracy, precisión, recall, F1 y ROC-AUC de prueba por encima de 0.60. La línea base de Jaccard queda impresa al lado. En este dataset las redacciones de un mismo grupo comparten casi todos los tokens, así que la línea base suele salir alta: sirve para leer la red con esa referencia. En una corrida de referencia con `SEED = 42` la línea base quedó en accuracy 0.922 y la red en accuracy 0.863, F1 0.873 y ROC-AUC 0.941. Esas tres cifras de la red pueden moverse un poco entre máquinas; el umbral que debe cumplirse es 0.60.

### Verificación

```python
assert accuracy > 0.60, f"Accuracy de prueba {accuracy:.3f}"
assert roc_auc > 0.60, f"ROC-AUC de prueba {roc_auc:.3f}"
assert 0.0 <= baseline_accuracy <= 1.0
print("✓ Evaluación en grupos no vistos completada")
```

---

## Paso 9: Tokens fuera de vocabulario

### Objetivo

Ver qué hace el sistema cuando una solicitud trae palabras que no estuvieron en el entrenamiento. Esas palabras comparten el índice `<UNK>` y el vector promedio construido en el paso 4.

`most_similar` de Word2Vec solo acepta palabras que ya están en el modelo. Llamarlo con una palabra nueva lanza `KeyError`. No es una estrategia de respaldo para fuera de vocabulario. Subword, FastText o un vocabulario más grande sí cubren ese caso; este laboratorio deja la palabra en `<UNK>`.

### Instrucciones

1. Mide el fuera de vocabulario real de los cortes que el modelo no usó para armar el índice:

```python
def tokens_desconocidos(parte):
    desconocidos = []
    for tokens in parte["tokens"]:
        desconocidos.extend(token for token in tokens if token not in word_to_idx)
    return desconocidos

for nombre, parte in (("val", df_val), ("test", df_test)):
    desconocidos = tokens_desconocidos(parte)
    total = sum(len(tokens) for tokens in parte["tokens"])
    print(f"{nombre}: {len(desconocidos)} tokens OOV de {total} ({len(desconocidos) / max(total, 1):.1%})")
    print(f"  Ejemplos: {sorted(set(desconocidos))[:12]}")
```

2. Comprueba el límite de Word2Vec y el índice que sí usa el modelo:

```python
palabra_nueva = "refinanciamiento"
try:
    w2v_model.wv.most_similar(palabra_nueva, topn=1)
    print("most_similar devolvió un vecino")
except KeyError:
    print(f"most_similar('{palabra_nueva}') lanzó KeyError: la palabra no está en Word2Vec.")

secuencia_oov = texto_a_secuencia(palabra_nueva + " blockchain", word_to_idx)
print(f"Secuencia de '{palabra_nueva} blockchain': {secuencia_oov}")
print(f"Índice UNK: {word_to_idx['<UNK>']}")
```

3. Predice un par con palabras que no están en las plantillas. La salida tiene que ser una probabilidad, aunque el modelo haya visto sobre todo `<UNK>`.

```python
prob = predecir_duplicado(
    "Solicito refinanciamiento hipotecario urgente",
    "Necesito reestructurar mi crédito de vivienda",
    best_model,
    word_to_idx,
    MAX_SEQ_LEN,
)
print(f"Probabilidad con palabras nuevas: {prob:.3f}")
print("El vector de <UNK> es el promedio de los embeddings de entrenamiento y permanece fijo.")
```

### Salida esperada

```text
most_similar('refinanciamiento') lanzó KeyError
Secuencia de 'refinanciamiento blockchain': [1, 1]
Probabilidad con palabras nuevas: un valor entre 0 y 1
```

### Verificación

```python
assert texto_a_secuencia("refinanciamiento blockchain", word_to_idx) == [1, 1]
assert 0.0 <= prob <= 1.0
print("✓ Las palabras nuevas caen en <UNK> y la predicción sigue definida")
```

---

## Paso 10: Resumen y cierre

### Objetivo

Dejar juntos los artefactos, las cifras de prueba y el criterio operativo. La pregunta final forma parte del laboratorio: hay que contestarla en una celda Markdown del notebook.

### Instrucciones

1. Imprime el resumen con los valores de esta ejecución:

```python
print("=" * 70)
print("RESUMEN - DETECCIÓN DE SOLICITUDES BANCARIAS DUPLICADAS")
print("=" * 70)
print(f"Directorio: {BASE_DIR}")
print(f"Solicitudes: {df.shape[0]} | grupos train/val/test: {len(grupos_train)}/{len(grupos_val)}/{len(grupos_test)}")
print(f"Word2Vec: {EMBEDDING_DIM}D, vocabulario {VOCAB_SIZE}, entrenado solo con train")
print(f"Pares train/val/test: {len(y_train)}/{len(y_val)}/{len(y_test)}")
print(f"Épocas ejecutadas: {len(history.history['loss'])}")
print(f"Línea base Jaccard accuracy/F1 en prueba: {baseline_accuracy:.3f} / {baseline_f1:.3f}")
print(f"Siamesa accuracy/precision/recall/F1/AUC en prueba: {accuracy:.3f} / {precision:.3f} / {recall:.3f} / {f1:.3f} / {roc_auc:.3f}")
print()
print("Lectura operativa:")
print(f"  Precision {precision:.3f}: de lo marcado como duplicado, esa fracción lo era.")
print("  Fusionar dos solicitudes distintas tiene costo de atención al cliente.")
print(f"  Recall {recall:.3f}: de los duplicados reales del corte de prueba, el modelo detectó esa fracción.")
print("  Dejar pasar un duplicado repite trabajo operativo.")
print()
print("Artefactos:")
for ruta in (
    os.path.join(RAW_DIR, "banking_requests.csv"),
    os.path.join(PROCESSED_DIR, "banking_requests_processed.csv"),
    os.path.join(EMBEDDINGS_DIR, "banking_w2v.model"),
    os.path.join(EMBEDDINGS_DIR, "embedding_matrix.npy"),
    os.path.join(EMBEDDINGS_DIR, "word_to_idx.json"),
    os.path.join(MODELS_DIR, "siamese_best.keras"),
    os.path.join(MODELS_DIR, "siamese_final.keras"),
    os.path.join(MODELS_DIR, "training_history.csv"),
):
    print(f"  {'✓' if os.path.exists(ruta) else '✗'} {ruta}")
```

2. En una celda Markdown, responde con tus cifras:

- ¿La red quedó por encima, cerca o por debajo de la línea base de Jaccard? ¿Qué parte del trabajo está haciendo el solapamiento de palabras?
- Si fusionar dos solicitudes distintas cuesta más que revisar un duplicado a mano, ¿subirías o bajarías el umbral de 0.5?
- ¿Qué palabra de un falso negativo cayó en `<UNK>` y cómo cambia eso la codificación de esa solicitud?

3. Verificación integral:

```python
comprobaciones = [
    ("Dataset generado", df.shape[0] >= 150 and df["grupo_duplicado"].nunique() == 50),
    ("Preprocesamiento en memoria", df["tokens"].map(lambda tokens: isinstance(tokens, list)).all()),
    ("Grupos sin intersección", grupos_train.isdisjoint(grupos_test) and grupos_train.isdisjoint(grupos_val)),
    ("Embeddings de entrenamiento", embedding_matrix.shape[1] == EMBEDDING_DIM),
    ("Pares de prueba", len(y_test) >= 20),
    ("Validación distinta de prueba", len(y_val) >= 20),
    ("Modelo entrenado", len(history.history["loss"]) > 5),
    ("Accuracy de prueba > 0.60", accuracy > 0.60),
    ("ROC-AUC de prueba > 0.60", roc_auc > 0.60),
    ("Checkpoint guardado", os.path.exists(os.path.join(MODELS_DIR, "siamese_best.keras"))),
    ("Predicción con OOV en [0, 1]", 0.0 <= prob <= 1.0),
]

todas = True
for nombre, resultado_ok in comprobaciones:
    print(f"  {'✓' if resultado_ok else '✗'} {nombre}")
    todas = todas and bool(resultado_ok)

print("✓ LABORATORIO COMPLETADO" if todas else "✗ REVISA EL PASO MARCADO")
assert todas
```

### Salida esperada

Todas las comprobaciones en `✓` y el mensaje `LABORATORIO COMPLETADO`.

---

## Resumen de entregables

| Artefacto | Ubicación | Descripción |
|-----------|-----------|-------------|
| Dataset crudo | `data\raw\banking_requests.csv` | Solicitudes sintéticas en UTF-8 con BOM |
| Dataset procesado | `data\processed\banking_requests_processed.csv` | `tokens` guardado como texto separado por espacios |
| Modelo Word2Vec | `embeddings\banking_w2v.model` | Embeddings 100D ajustados solo con grupos de entrenamiento |
| Matriz de embeddings | `embeddings\embedding_matrix.npy` | `float32`, fila 0 en ceros, fila 1 con el vector promedio |
| Vocabulario | `embeddings\word_to_idx.json` | Palabra → índice, UTF-8 |
| Mejor modelo | `models\siamese_best.keras` | Checkpoint de menor `val_loss` |
| Modelo final | `models\siamese_final.keras` | Pesos restaurados al mejor `val_loss` |
| Historial | `models\training_history.csv` | Pérdida y accuracy de entrenamiento y validación |
| Notebook | `notebooks\siamese_banking_duplicates.ipynb` | Celdas ejecutadas en Visual Studio Code |
| Gráficos | `notebooks\*.png` | EDA, curvas y métricas de prueba |

## Criterios de evaluación

| Criterio | Peso | Descripción |
|----------|------|-------------|
| Pipeline de preprocesamiento | 20% | Normalización de español bancario, números conservados y tokens utilizables |
| Embeddings Word2Vec | 15% | Ajustados solo con entrenamiento y matriz compatible con Keras, con PAD y UNK definidos |
| Generación de pares | 15% | Clases equilibradas y ningún grupo compartido entre entrenamiento, validación y prueba |
| Arquitectura siamesa | 20% | LSTM unidireccional compartido, distancia L1 serializable y embeddings congelados |
| Entrenamiento con callbacks | 15% | Early stopping, checkpoint y learning rate mirando la validación; la prueba no entra a `fit` |
| Evaluación e interpretación | 15% | Métricas de prueba, comparación con Jaccard y lectura del umbral en costo operativo |

## Qué queda fuera de esta práctica

Ampliar el corpus con solicitudes reales anonimizadas, pasar a FastText o a un modelo con subwords, probar un Bi-LSTM y elegir el umbral con el costo de un falso positivo frente a un falso negativo. Otra práctica del curso cubre la extracción de entidades; aquí la categoría solo sirve para estratificar los grupos.
