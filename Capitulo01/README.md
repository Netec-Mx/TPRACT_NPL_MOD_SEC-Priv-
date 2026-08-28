# Construcción de un sistema para analizar y detectar solicitudes bancarias duplicadas

## Metadatos

| Campo | Valor |
|-------|-------|
| **Duración** | 105 minutos |
| **Complejidad** | Alta |
| **Nivel Bloom** | Crear |

## Descripción General

En esta práctica integradora construirás desde cero un sistema completo de detección de solicitudes bancarias duplicadas. Partirás de un dataset sintético de solicitudes en español, realizarás análisis exploratorio, preprocesarás el texto con técnicas de NLP, entrenarás embeddings Word2Vec, generarás pares de entrenamiento, diseñarás e implementarás una red neuronal siamesa con capas LSTM compartidas, y finalmente evaluarás el sistema con métricas de clasificación binaria. Al finalizar dispondrás de un modelo funcional capaz de identificar si dos solicitudes bancarias son duplicadas.

## Objetivos de Aprendizaje

- [ ] Implementar un pipeline completo de preprocesamiento NLP para textos bancarios en español (tokenización, normalización, eliminación de stopwords, padding)
- [ ] Entrenar embeddings Word2Vec sobre el corpus bancario y construir una matriz de pesos compatible con Keras
- [ ] Diseñar e implementar una red neuronal siamesa con LSTM compartido para clasificar pares de solicitudes
- [ ] Entrenar el modelo siamés monitoreando métricas de validación con callbacks apropiados
- [ ] Evaluar el sistema con accuracy, precisión, recall, F1-score y ROC-AUC, interpretando resultados en contexto bancario

## Prerrequisitos

### Conocimientos Requeridos

| Tema | Nivel |
|------|-------|
| Python 3.10+ y JupyterLab | Intermedio |
| NumPy y pandas | Intermedio |
| Redes neuronales recurrentes (LSTM) | Conceptual sólido |
| Redes siamesas y funciones de pérdida | Conceptual |
| Word embeddings (Word2Vec) | Conceptual |
| Keras/TensorFlow (ciclo train/eval) | Básico-Intermedio |

### Acceso Requerido

- Terminal con permisos de escritura en el directorio de trabajo
- Python 3.10.12 instalado
- Conexión a Internet (para descarga de recursos NLTK)
- Mínimo 8 GB RAM disponible

## Entorno del Laboratorio

### Hardware Mínimo

| Componente | Especificación |
|------------|---------------|
| CPU | Intel Core i5 8ª gen / AMD Ryzen 5 3000+ (4 núcleos) |
| RAM | 8 GB DDR4 (recomendado 16 GB) |
| Almacenamiento | 5 GB libres |
| GPU (opcional) | NVIDIA CUDA 11.8+, 4 GB VRAM |

### Software

| Herramienta | Versión |
|-------------|---------|
| Python | 3.10.12 |
| TensorFlow / Keras | 2.15.0 |
| NumPy | 1.26.4 |
| pandas | 2.2.1 |
| scikit-learn | 1.4.1.post1 |
| NLTK | 3.8.1 |
| Gensim | 4.3.2 |
| Matplotlib / seaborn | 3.8.3 / 0.13.2 |
| JupyterLab | 4.1.5 |

### Configuración Inicial del Entorno

Ejecuta los siguientes comandos en la terminal para preparar el entorno completo:

```bash
# Crear directorio raíz del proyecto
mkdir -p ~/nlp_banking_lab/{data/raw,data/processed,models,notebooks,embeddings}
cd ~/nlp_banking_lab

# Crear entorno virtual
python3 -m venv nlp_env
source nlp_env/bin/activate

# Crear requirements.txt
cat > requirements.txt << 'EOF'
tensorflow==2.15.0
keras==2.15.0
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.1.post1
nltk==3.8.1
gensim==4.3.2
matplotlib==3.8.3
seaborn==0.13.2
tqdm==4.66.2
Unidecode==1.3.8
jupyterlab==4.1.5
EOF

# Instalar dependencias
pip install --upgrade pip==24.0
pip install -r requirements.txt

# Lanzar JupyterLab
jupyter lab --port=8888
```

Abre el navegador en `http://localhost:8888/lab` y crea un nuevo notebook en `notebooks/` con el nombre `siamese_banking_duplicates.ipynb`.

---

## Paso 1: Generación del Dataset Sintético de Solicitudes Bancarias

### Objetivo

Crear un dataset realista de solicitudes bancarias en español con grupos de duplicados identificados, que servirá como base para todo el laboratorio.

### Instrucciones

1. En la primera celda del notebook, importa las librerías necesarias y configura las semillas de reproducibilidad:

```python
import numpy as np
import pandas as pd
import random
import os

# Reproducibilidad
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Directorio base
BASE_DIR = os.path.expanduser("~/nlp_banking_lab")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")

print(f"Directorio base: {BASE_DIR}")
```

2. Genera el dataset sintético con solicitudes bancarias agrupadas por duplicados:

```python
# Plantillas de solicitudes bancarias por categoría
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

# Variaciones para generar duplicados semánticos
variaciones = [
    lambda t: t.replace("Solicito", "Pido").replace("solicito", "pido"),
    lambda t: t.replace("Necesito", "Requiero").replace("necesito", "requiero"),
    lambda t: t.replace("Quiero", "Deseo").replace("quiero", "deseo"),
    lambda t: t + " por favor",
    lambda t: "Buenos días, " + t[0].lower() + t[1:],
    lambda t: t + ". Agradezco su pronta respuesta",
    lambda t: "Hola, " + t[0].lower() + t[1:],
    lambda t: t.replace("mi", "mi ").replace("  ", " "),  # mantener igual
    lambda t: "Urgente: " + t[0].lower() + t[1:],
    lambda t: t + ". Quedo atento",
]

# Generar dataset
records = []
id_counter = 0
grupo_counter = 0

for categoria, textos_base in templates.items():
    for texto_base in textos_base:
        grupo_counter += 1
        # Texto original
        id_counter += 1
        records.append({
            "id": id_counter,
            "texto_solicitud": texto_base,
            "categoria": categoria,
            "grupo_duplicado": grupo_counter
        })
        # Generar 2-4 variaciones (duplicados semánticos)
        n_variaciones = random.randint(2, 4)
        selected_vars = random.sample(variaciones, n_variaciones)
        for var_func in selected_vars:
            id_counter += 1
            texto_variado = var_func(texto_base)
            records.append({
                "id": id_counter,
                "texto_solicitud": texto_variado,
                "categoria": categoria,
                "grupo_duplicado": grupo_counter
            })

df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Guardar dataset
output_path = os.path.join(RAW_DIR, "banking_requests.csv")
df.to_csv(output_path, index=False)
print(f"Dataset generado: {df.shape[0]} solicitudes, {df['grupo_duplicado'].nunique()} grupos")
print(f"Guardado en: {output_path}")
df.head(10)
```

### Salida Esperada

```
Dataset generado: ~185 solicitudes, 50 grupos
Guardado en: /home/usuario/nlp_banking_lab/data/raw/banking_requests.csv
```

Se mostrará una tabla con las primeras 10 filas conteniendo columnas `id`, `texto_solicitud`, `categoria` y `grupo_duplicado`.

### Verificación

```python
assert df.shape[0] > 150, "Dataset debe tener más de 150 registros"
assert set(df.columns) == {"id", "texto_solicitud", "categoria", "grupo_duplicado"}
assert df["categoria"].nunique() == 5
print("✓ Dataset generado correctamente")
```

---

## Paso 2: Análisis Exploratorio de Datos (EDA)

### Objetivo

Comprender la distribución del dataset, la longitud de los textos y el balance de clases para informar decisiones de preprocesamiento y modelado.

### Instrucciones

1. Analiza la distribución de categorías y grupos de duplicados:

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribución de categorías
cat_counts = df["categoria"].value_counts()
sns.barplot(x=cat_counts.index, y=cat_counts.values, ax=axes[0], palette="viridis")
axes[0].set_title("Distribución de Categorías de Solicitudes")
axes[0].set_xlabel("Categoría")
axes[0].set_ylabel("Cantidad")
axes[0].tick_params(axis='x', rotation=45)

# Distribución de solicitudes por grupo de duplicados
grupo_sizes = df.groupby("grupo_duplicado").size()
sns.histplot(grupo_sizes, bins=10, ax=axes[1], color="steelblue")
axes[1].set_title("Tamaño de Grupos de Duplicados")
axes[1].set_xlabel("Solicitudes por grupo")
axes[1].set_ylabel("Frecuencia")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebooks", "eda_distribucion.png"), dpi=100)
plt.show()

print(f"\nEstadísticas de grupos de duplicados:")
print(f"  Tamaño mínimo: {grupo_sizes.min()}")
print(f"  Tamaño máximo: {grupo_sizes.max()}")
print(f"  Tamaño promedio: {grupo_sizes.mean():.1f}")
```

2. Analiza la longitud de los textos:

```python
df["longitud_chars"] = df["texto_solicitud"].str.len()
df["longitud_palabras"] = df["texto_solicitud"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df["longitud_palabras"], bins=20, ax=axes[0], color="coral")
axes[0].set_title("Distribución de Longitud (palabras)")
axes[0].set_xlabel("Número de palabras")
axes[0].set_ylabel("Frecuencia")

sns.boxplot(data=df, x="categoria", y="longitud_palabras", ax=axes[1], palette="Set2")
axes[1].set_title("Longitud por Categoría")
axes[1].set_xlabel("Categoría")
axes[1].set_ylabel("Palabras")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebooks", "eda_longitudes.png"), dpi=100)
plt.show()

print(f"\nEstadísticas de longitud (palabras):")
print(f"  Mínimo: {df['longitud_palabras'].min()}")
print(f"  Máximo: {df['longitud_palabras'].max()}")
print(f"  Promedio: {df['longitud_palabras'].mean():.1f}")
print(f"  Percentil 95: {df['longitud_palabras'].quantile(0.95):.0f}")

# Definir longitud máxima para padding
MAX_SEQ_LEN = int(df['longitud_palabras'].quantile(0.95)) + 2
print(f"\n  MAX_SEQ_LEN seleccionado: {MAX_SEQ_LEN}")
```

### Salida Esperada

Se generarán dos figuras con gráficos de distribución. La longitud promedio estará entre 8-14 palabras por solicitud y el `MAX_SEQ_LEN` será aproximadamente 15-18.

### Verificación

```python
assert "longitud_palabras" in df.columns
assert MAX_SEQ_LEN > 10, "MAX_SEQ_LEN debe ser mayor a 10"
print("✓ EDA completado correctamente")
```

---

## Paso 3: Preprocesamiento de Texto NLP

### Objetivo

Implementar un pipeline de limpieza y normalización de texto adaptado al dominio bancario en español, incluyendo tokenización, eliminación de stopwords y normalización.

### Instrucciones

1. Descarga los recursos NLTK necesarios e implementa el pipeline de preprocesamiento:

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import re

# Descargar recursos NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Stopwords en español - personalizadas para dominio bancario
stop_words_es = set(stopwords.words('spanish'))
# Mantener palabras relevantes para el dominio bancario
palabras_relevantes = {"no", "sin", "contra", "sobre", "mi", "mis"}
stop_words_es = stop_words_es - palabras_relevantes

print(f"Stopwords cargadas: {len(stop_words_es)} palabras")
print(f"Ejemplo: {list(stop_words_es)[:10]}")
```

2. Define la función de preprocesamiento:

```python
def preprocesar_texto(texto):
    """
    Pipeline de preprocesamiento para solicitudes bancarias en español.
    
    Pasos:
    1. Convertir a minúsculas
    2. Eliminar acentos (normalización)
    3. Eliminar caracteres especiales y números
    4. Tokenizar
    5. Eliminar stopwords
    6. Filtrar tokens muy cortos (< 2 caracteres)
    
    Args:
        texto (str): Texto de la solicitud bancaria
        
    Returns:
        list: Lista de tokens preprocesados
    """
    if not isinstance(texto, str):
        return []
    
    # 1. Minúsculas
    texto = texto.lower()
    
    # 2. Eliminar acentos
    texto = unidecode(texto)
    
    # 3. Eliminar caracteres especiales y números
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    
    # 4. Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    # 5. Tokenizar
    tokens = texto.split()
    
    # 6. Eliminar stopwords y tokens cortos
    tokens = [t for t in tokens if t not in stop_words_es and len(t) >= 2]
    
    return tokens

# Probar con ejemplo
ejemplo = "Solicito información sobre créditos de libre inversión disponibles"
resultado = preprocesar_texto(ejemplo)
print(f"Original: {ejemplo}")
print(f"Procesado: {resultado}")
```

3. Aplica el preprocesamiento a todo el dataset:

```python
from tqdm import tqdm

# Aplicar preprocesamiento
tqdm.pandas(desc="Preprocesando textos")
df["tokens"] = df["texto_solicitud"].progress_apply(preprocesar_texto)
df["texto_limpio"] = df["tokens"].apply(lambda t: " ".join(t))

# Estadísticas post-preprocesamiento
df["n_tokens"] = df["tokens"].apply(len)
print(f"\nEstadísticas post-preprocesamiento:")
print(f"  Tokens promedio: {df['n_tokens'].mean():.1f}")
print(f"  Tokens mínimo: {df['n_tokens'].min()}")
print(f"  Tokens máximo: {df['n_tokens'].max()}")

# Vocabulario
all_tokens = [token for tokens in df["tokens"] for token in tokens]
vocabulario = sorted(set(all_tokens))
print(f"  Vocabulario total: {len(vocabulario)} palabras únicas")

# Guardar dataset procesado
processed_path = os.path.join(PROCESSED_DIR, "banking_requests_processed.csv")
df.to_csv(processed_path, index=False)
print(f"\nDataset procesado guardado en: {processed_path}")

# Mostrar ejemplos
df[["texto_solicitud", "texto_limpio", "n_tokens"]].head(5)
```

### Salida Esperada

```
Stopwords cargadas: ~308 palabras
Original: Solicito información sobre créditos de libre inversión disponibles
Procesado: ['solicito', 'informacion', 'sobre', 'creditos', 'libre', 'inversion', 'disponibles']

Estadísticas post-preprocesamiento:
  Tokens promedio: ~6.5
  Tokens mínimo: 3
  Tokens máximo: ~12
  Vocabulario total: ~120 palabras únicas
```

### Verificación

```python
assert "tokens" in df.columns
assert "texto_limpio" in df.columns
assert df["n_tokens"].min() >= 2, "Todos los textos deben tener al menos 2 tokens"
assert len(vocabulario) > 50, "Vocabulario debe tener más de 50 palabras"
print("✓ Preprocesamiento completado correctamente")
```

---

## Paso 4: Entrenamiento de Embeddings Word2Vec

### Objetivo

Entrenar un modelo Word2Vec sobre el corpus bancario para obtener representaciones vectoriales densas de las palabras del dominio, y construir una matriz de embeddings compatible con la capa Embedding de Keras.

### Instrucciones

1. Entrena el modelo Word2Vec con Gensim:

```python
from gensim.models import Word2Vec

# Parámetros del modelo
EMBEDDING_DIM = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 1  # Incluir todas las palabras dado el tamaño del corpus
W2V_EPOCHS = 100   # Más épocas por corpus pequeño
W2V_WORKERS = 4

# Preparar corpus (lista de listas de tokens)
corpus = df["tokens"].tolist()

print(f"Entrenando Word2Vec...")
print(f"  Corpus: {len(corpus)} documentos")
print(f"  Dimensión embeddings: {EMBEDDING_DIM}")
print(f"  Ventana: {W2V_WINDOW}")
print(f"  Épocas: {W2V_EPOCHS}")

# Entrenar modelo
w2v_model = Word2Vec(
    sentences=corpus,
    vector_size=EMBEDDING_DIM,
    window=W2V_WINDOW,
    min_count=W2V_MIN_COUNT,
    workers=W2V_WORKERS,
    epochs=W2V_EPOCHS,
    seed=SEED,
    sg=1  # Skip-gram (mejor para corpus pequeños)
)

# Guardar modelo
w2v_path = os.path.join(EMBEDDINGS_DIR, "banking_w2v.model")
w2v_model.save(w2v_path)
print(f"\nModelo guardado en: {w2v_path}")
print(f"Vocabulario del modelo: {len(w2v_model.wv)} palabras")
```

2. Explora los embeddings entrenados:

```python
# Verificar palabras similares
test_words = ["credito", "cuenta", "tarjeta", "cancelar", "cobro"]

for word in test_words:
    if word in w2v_model.wv:
        similares = w2v_model.wv.most_similar(word, topn=5)
        print(f"\nPalabras similares a '{word}':")
        for sim_word, score in similares:
            print(f"  {sim_word}: {score:.3f}")
    else:
        print(f"\n'{word}' no está en el vocabulario")
```

3. Construye el vocabulario indexado y la matriz de embeddings para Keras:

```python
# Construir vocabulario indexado
# Índice 0 reservado para padding, índice 1 para token desconocido (UNK)
word_to_idx = {"<PAD>": 0, "<UNK>": 1}
idx_to_word = {0: "<PAD>", 1: "<UNK>"}

for idx, word in enumerate(vocabulario, start=2):
    word_to_idx[word] = idx
    idx_to_word[idx] = word

VOCAB_SIZE = len(word_to_idx)
print(f"Tamaño del vocabulario (con PAD y UNK): {VOCAB_SIZE}")

# Construir matriz de embeddings
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

palabras_con_embedding = 0
palabras_sin_embedding = 0

for word, idx in word_to_idx.items():
    if word in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[word]
        palabras_con_embedding += 1
    elif word not in ["<PAD>", "<UNK>"]:
        # Inicialización aleatoria para palabras sin embedding
        embedding_matrix[idx] = np.random.normal(0, 0.1, EMBEDDING_DIM)
        palabras_sin_embedding += 1

print(f"Palabras con embedding W2V: {palabras_con_embedding}")
print(f"Palabras sin embedding (aleatorio): {palabras_sin_embedding}")
print(f"Forma de la matriz: {embedding_matrix.shape}")

# Guardar artefactos
np.save(os.path.join(EMBEDDINGS_DIR, "embedding_matrix.npy"), embedding_matrix)

import json
with open(os.path.join(EMBEDDINGS_DIR, "word_to_idx.json"), "w") as f:
    json.dump(word_to_idx, f, ensure_ascii=False)

print("✓ Matriz de embeddings y vocabulario guardados")
```

### Salida Esperada

```
Entrenando Word2Vec...
  Corpus: ~185 documentos
  Dimensión embeddings: 100
  Ventana: 5
  Épocas: 100

Modelo guardado en: /home/usuario/nlp_banking_lab/embeddings/banking_w2v.model
Vocabulario del modelo: ~120 palabras

Tamaño del vocabulario (con PAD y UNK): ~122
Palabras con embedding W2V: ~120
Palabras sin embedding (aleatorio): 0
Forma de la matriz: (122, 100)
```

### Verificación

```python
assert embedding_matrix.shape == (VOCAB_SIZE, EMBEDDING_DIM)
assert embedding_matrix[0].sum() == 0.0, "PAD debe ser vector cero"
assert word_to_idx["<PAD>"] == 0
assert word_to_idx["<UNK>"] == 1
print("✓ Embeddings Word2Vec construidos correctamente")
```

---

## Paso 5: Generación de Pares de Entrenamiento

### Objetivo

Crear pares de solicitudes etiquetados como duplicados (label=1) o no duplicados (label=0) para entrenar la red siamesa, asegurando balance de clases.

### Instrucciones

1. Genera pares positivos (duplicados) y negativos (no duplicados):

```python
from itertools import combinations

def generar_pares(dataframe, seed=42):
    """
    Genera pares balanceados de solicitudes duplicadas y no duplicadas.
    
    Pares positivos: solicitudes del mismo grupo_duplicado
    Pares negativos: solicitudes de diferentes grupos (muestreados)
    
    Returns:
        pairs_left: lista de textos izquierdos
        pairs_right: lista de textos derechos
        labels: lista de etiquetas (1=duplicado, 0=no duplicado)
    """
    rng = np.random.RandomState(seed)
    
    pairs_left = []
    pairs_right = []
    labels = []
    
    # Pares positivos: combinaciones dentro de cada grupo
    grupos = dataframe.groupby("grupo_duplicado")["texto_limpio"].apply(list)
    
    for grupo_id, textos in grupos.items():
        if len(textos) >= 2:
            for t1, t2 in combinations(textos, 2):
                pairs_left.append(t1)
                pairs_right.append(t2)
                labels.append(1)
    
    n_positivos = len(labels)
    print(f"Pares positivos generados: {n_positivos}")
    
    # Pares negativos: muestrear pares de diferentes grupos
    all_texts_by_group = grupos.to_dict()
    group_ids = list(all_texts_by_group.keys())
    
    n_negativos_target = n_positivos  # Balance 1:1
    neg_count = 0
    
    while neg_count < n_negativos_target:
        g1, g2 = rng.choice(group_ids, size=2, replace=False)
        t1 = rng.choice(all_texts_by_group[g1])
        t2 = rng.choice(all_texts_by_group[g2])
        pairs_left.append(t1)
        pairs_right.append(t2)
        labels.append(0)
        neg_count += 1
    
    print(f"Pares negativos generados: {neg_count}")
    print(f"Total de pares: {len(labels)}")
    
    return pairs_left, pairs_right, np.array(labels)

# Generar pares
pairs_left, pairs_right, labels = generar_pares(df, seed=SEED)

print(f"\nDistribución de etiquetas:")
print(f"  Duplicados (1): {(labels == 1).sum()}")
print(f"  No duplicados (0): {(labels == 0).sum()}")
print(f"  Ratio: {(labels == 1).sum() / len(labels):.2%}")
```

2. Convierte los textos a secuencias numéricas con padding:

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

def texto_a_secuencia(texto, word_to_idx, max_len):
    """Convierte texto preprocesado a secuencia de índices con padding."""
    tokens = texto.split()
    secuencia = [word_to_idx.get(t, word_to_idx["<UNK>"]) for t in tokens]
    return secuencia

# Convertir todos los pares a secuencias
seqs_left = [texto_a_secuencia(t, word_to_idx, MAX_SEQ_LEN) for t in pairs_left]
seqs_right = [texto_a_secuencia(t, word_to_idx, MAX_SEQ_LEN) for t in pairs_right]

# Aplicar padding
X_left = pad_sequences(seqs_left, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
X_right = pad_sequences(seqs_right, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

print(f"Forma X_left: {X_left.shape}")
print(f"Forma X_right: {X_right.shape}")
print(f"MAX_SEQ_LEN: {MAX_SEQ_LEN}")

# Ejemplo de secuencia
print(f"\nEjemplo texto: '{pairs_left[0]}'")
print(f"Secuencia: {X_left[0]}")
```

3. Divide en conjuntos de entrenamiento y prueba:

```python
from sklearn.model_selection import train_test_split

X_left_train, X_left_test, X_right_train, X_right_test, y_train, y_test = train_test_split(
    X_left, X_right, labels,
    test_size=0.2,
    random_state=SEED,
    stratify=labels
)

print(f"Conjunto de entrenamiento: {len(y_train)} pares")
print(f"  Positivos: {(y_train == 1).sum()}, Negativos: {(y_train == 0).sum()}")
print(f"Conjunto de prueba: {len(y_test)} pares")
print(f"  Positivos: {(y_test == 1).sum()}, Negativos: {(y_test == 0).sum()}")
```

### Salida Esperada

```
Pares positivos generados: ~280
Pares negativos generados: ~280
Total de pares: ~560

Forma X_left: (560, 16)
Forma X_right: (560, 16)

Conjunto de entrenamiento: ~448 pares
Conjunto de prueba: ~112 pares
```

### Verificación

```python
assert X_left_train.shape[1] == MAX_SEQ_LEN
assert X_right_train.shape[1] == MAX_SEQ_LEN
assert len(y_train) == len(X_left_train)
assert abs((y_train == 1).sum() - (y_train == 0).sum()) < 5, "Clases deben estar balanceadas"
print("✓ Pares de entrenamiento generados correctamente")
```

---

## Paso 6: Diseño e Implementación de la Red Neuronal Siamesa

### Objetivo

Implementar una red neuronal siamesa con LSTM compartido que reciba dos solicitudes y prediga si son duplicadas, utilizando la API funcional de Keras.

### Instrucciones

1. Define la arquitectura de la red siamesa:

```python
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout,
    Lambda, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Fijar semilla para TensorFlow
tf.random.set_seed(SEED)

def crear_red_siamesa(vocab_size, embedding_dim, embedding_matrix, max_seq_len, lstm_units=64):
    """
    Crea una red neuronal siamesa con LSTM compartido.
    
    Arquitectura:
    - Capa Embedding (pesos pre-entrenados Word2Vec)
    - LSTM bidireccional compartido
    - Cálculo de distancia Manhattan (L1)
    - Capa densa de salida con sigmoid
    
    Args:
        vocab_size: tamaño del vocabulario
        embedding_dim: dimensión de los embeddings
        embedding_matrix: matriz de pesos pre-entrenados
        max_seq_len: longitud máxima de secuencia
        lstm_units: unidades en la capa LSTM
        
    Returns:
        model: modelo Keras compilado
    """
    # Inputs
    input_left = Input(shape=(max_seq_len,), name="input_left")
    input_right = Input(shape=(max_seq_len,), name="input_right")
    
    # Capa Embedding compartida (pesos pre-entrenados)
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_seq_len,
        trainable=True,  # Fine-tuning de embeddings
        mask_zero=True,
        name="shared_embedding"
    )
    
    # LSTM compartido
    shared_lstm = LSTM(
        units=lstm_units,
        return_sequences=False,
        dropout=0.2,
        recurrent_dropout=0.2,
        name="shared_lstm"
    )
    
    # Procesar ambas entradas con las mismas capas
    embedded_left = embedding_layer(input_left)
    embedded_right = embedding_layer(input_right)
    
    encoded_left = shared_lstm(embedded_left)
    encoded_right = shared_lstm(embedded_right)
    
    # Calcular distancia Manhattan (L1)
    l1_distance = Lambda(
        lambda tensors: K.abs(tensors[0] - tensors[1]),
        name="l1_distance"
    )([encoded_left, encoded_right])
    
    # Capas densas de clasificación
    x = BatchNormalization()(l1_distance)
    x = Dense(64, activation='relu', name="dense_1")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', name="dense_2")(x)
    x = Dropout(0.2)(x)
    
    # Salida: probabilidad de ser duplicado
    output = Dense(1, activation='sigmoid', name="output")(x)
    
    # Construir modelo
    model = Model(inputs=[input_left, input_right], outputs=output, name="siamese_lstm")
    
    # Compilar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Crear modelo
model = crear_red_siamesa(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    embedding_matrix=embedding_matrix,
    max_seq_len=MAX_SEQ_LEN,
    lstm_units=64
)

# Resumen del modelo
model.summary()
```

2. Visualiza la arquitectura:

```python
# Contar parámetros por capa
print("\nDetalle de parámetros:")
print(f"  Embedding: {VOCAB_SIZE * EMBEDDING_DIM:,} parámetros")
print(f"  LSTM: {4 * ((EMBEDDING_DIM + 64) * 64 + 64):,} parámetros (aprox)")
print(f"  Total trainable: {model.count_params():,}")

# Guardar diagrama de arquitectura (opcional, requiere graphviz)
try:
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(BASE_DIR, "notebooks", "siamese_architecture.png"),
        show_shapes=True,
        show_layer_names=True
    )
    print("✓ Diagrama guardado en notebooks/siamese_architecture.png")
except Exception as e:
    print(f"Nota: No se pudo generar diagrama visual ({e})")
    print("  Esto no afecta el entrenamiento del modelo.")
```

### Salida Esperada

El resumen del modelo mostrará:
- 2 capas Input (max_seq_len,)
- 1 capa Embedding compartida
- 1 capa LSTM compartida (64 unidades)
- Capa Lambda para distancia L1
- Capas Dense (64 → 32 → 1)
- Total de parámetros: ~60,000-80,000

### Verificación

```python
assert model.input_shape == [(None, MAX_SEQ_LEN), (None, MAX_SEQ_LEN)]
assert model.output_shape == (None, 1)
assert len(model.layers) > 5
print("✓ Red siamesa construida correctamente")
```

---

## Paso 7: Entrenamiento del Modelo Siamés

### Objetivo

Entrenar la red siamesa con los pares generados, utilizando callbacks para monitorear el rendimiento y prevenir sobreajuste.

### Instrucciones

1. Configura los callbacks y entrena el modelo:

```python
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, "siamese_best.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
]

# Parámetros de entrenamiento
BATCH_SIZE = 32
EPOCHS = 100

print("Iniciando entrenamiento...")
print(f"  Épocas máximas: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Muestras entrenamiento: {len(y_train)}")
print(f"  Muestras validación: {len(y_test)}")
print("-" * 50)

# Entrenar
history = model.fit(
    x=[X_left_train, X_right_train],
    y=y_train,
    validation_data=([X_left_test, X_right_test], y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print(f"\nEntrenamiento finalizado en época {len(history.history['loss'])}")
print(f"Mejor val_accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Mejor val_loss: {min(history.history['val_loss']):.4f}")
```

2. Visualiza las curvas de entrenamiento:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Curva de pérdida
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_title('Evolución de la Pérdida (Binary Crossentropy)')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Pérdida')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Curva de accuracy
axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[1].set_title('Evolución del Accuracy')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebooks", "training_curves.png"), dpi=100)
plt.show()
```

3. Guarda el modelo final:

```python
# Guardar modelo completo
final_model_path = os.path.join(MODELS_DIR, "siamese_final.keras")
model.save(final_model_path)
print(f"Modelo final guardado en: {final_model_path}")

# Guardar historial de entrenamiento
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(MODELS_DIR, "training_history.csv"), index=False)
print("Historial de entrenamiento guardado")
```

### Salida Esperada

```
Iniciando entrenamiento...
  Épocas máximas: 100
  Batch size: 32
  Muestras entrenamiento: ~448
  Muestras validación: ~112

Entrenamiento finalizado en época ~40-60
Mejor val_accuracy: ~0.80-0.92
Mejor val_loss: ~0.30-0.50
```

Las curvas de entrenamiento deben mostrar convergencia sin divergencia significativa entre train y val (indicando que no hay sobreajuste severo).

### Verificación

```python
assert len(history.history['loss']) > 5, "Modelo debe entrenar más de 5 épocas"
assert max(history.history['val_accuracy']) > 0.65, "Val accuracy debe superar 0.65"
assert os.path.exists(os.path.join(MODELS_DIR, "siamese_best.keras"))
print("✓ Entrenamiento completado correctamente")
```

---

## Paso 8: Evaluación del Modelo con Métricas de Clasificación

### Objetivo

Evaluar exhaustivamente el modelo siamés con métricas estándar de clasificación binaria (accuracy, precision, recall, F1-score, ROC-AUC) y analizar los resultados en contexto bancario.

### Instrucciones

1. Genera predicciones sobre el conjunto de prueba:

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve
)

# Cargar mejor modelo
best_model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "siamese_best.keras")
)

# Predicciones
y_pred_proba = best_model.predict([X_left_test, X_right_test]).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

print("Predicciones generadas:")
print(f"  Rango probabilidades: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
print(f"  Predicciones positivas: {y_pred.sum()}")
print(f"  Predicciones negativas: {(1 - y_pred).sum()}")
```

2. Calcula y muestra todas las métricas:

```python
# Métricas principales
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("=" * 60)
print("MÉTRICAS DE EVALUACIÓN - DETECCIÓN DE DUPLICADOS BANCARIOS")
print("=" * 60)
print(f"\n  Accuracy:  {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"  Precision: {precision:.4f}  ({precision*100:.1f}%)")
print(f"  Recall:    {recall:.4f}  ({recall*100:.1f}%)")
print(f"  F1-Score:  {f1:.4f}  ({f1*100:.1f}%)")
print(f"  ROC-AUC:   {roc_auc:.4f}  ({roc_auc*100:.1f}%)")

print("\n" + "-" * 60)
print("\nReporte de Clasificación Completo:")
print(classification_report(
    y_test, y_pred,
    target_names=["No Duplicado", "Duplicado"],
    digits=4
))
```

3. Visualiza la matriz de confusión y la curva ROC:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=["No Duplicado", "Duplicado"],
    yticklabels=["No Duplicado", "Duplicado"],
    ax=axes[0]
)
axes[0].set_title("Matriz de Confusión")
axes[0].set_xlabel("Predicción")
axes[0].set_ylabel("Real")

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Aleatorio')
axes[1].set_title("Curva ROC")
axes[1].set_xlabel("Tasa de Falsos Positivos (FPR)")
axes[1].set_ylabel("Tasa de Verdaderos Positivos (TPR)")
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebooks", "evaluation_metrics.png"), dpi=100)
plt.show()
```

4. Análisis de errores y prueba con ejemplos nuevos:

```python
# Análisis de errores
print("\n" + "=" * 60)
print("ANÁLISIS DE ERRORES")
print("=" * 60)

# Falsos negativos (duplicados no detectados)
fn_mask = (y_test == 1) & (y_pred == 0)
fn_indices = np.where(fn_mask)[0]
print(f"\nFalsos Negativos (duplicados no detectados): {fn_mask.sum()}")
if fn_mask.sum() > 0:
    for i in fn_indices[:3]:
        left_text = " ".join([idx_to_word.get(idx, "?") for idx in X_left_test[i] if idx != 0])
        right_text = " ".join([idx_to_word.get(idx, "?") for idx in X_right_test[i] if idx != 0])
        print(f"  Par {i}: prob={y_pred_proba[i]:.3f}")
        print(f"    Izq: {left_text}")
        print(f"    Der: {right_text}")

# Falsos positivos (no duplicados marcados como duplicados)
fp_mask = (y_test == 0) & (y_pred == 1)
fp_indices = np.where(fp_mask)[0]
print(f"\nFalsos Positivos (no duplicados marcados como duplicados): {fp_mask.sum()}")
if fp_mask.sum() > 0:
    for i in fp_indices[:3]:
        left_text = " ".join([idx_to_word.get(idx, "?") for idx in X_left_test[i] if idx != 0])
        right_text = " ".join([idx_to_word.get(idx, "?") for idx in X_right_test[i] if idx != 0])
        print(f"  Par {i}: prob={y_pred_proba[i]:.3f}")
        print(f"    Izq: {left_text}")
        print(f"    Der: {right_text}")
```

5. Prueba el modelo con nuevos ejemplos:

```python
def predecir_duplicado(texto1, texto2, model, word_to_idx, max_len):
    """
    Predice si dos solicitudes son duplicadas.
    
    Args:
        texto1, texto2: textos de solicitudes bancarias
        model: modelo siamés entrenado
        word_to_idx: diccionario de vocabulario
        max_len: longitud máxima de secuencia
        
    Returns:
        probabilidad de ser duplicado
    """
    # Preprocesar
    tokens1 = preprocesar_texto(texto1)
    tokens2 = preprocesar_texto(texto2)
    
    # Convertir a secuencias
    seq1 = [word_to_idx.get(t, word_to_idx["<UNK>"]) for t in tokens1]
    seq2 = [word_to_idx.get(t, word_to_idx["<UNK>"]) for t in tokens2]
    
    # Padding
    seq1 = pad_sequences([seq1], maxlen=max_len, padding='post', truncating='post')
    seq2 = pad_sequences([seq2], maxlen=max_len, padding='post', truncating='post')
    
    # Predecir
    prob = model.predict([seq1, seq2], verbose=0)[0][0]
    return prob

# Ejemplos de prueba
print("\n" + "=" * 60)
print("PRUEBAS CON NUEVOS EJEMPLOS")
print("=" * 60)

test_cases = [
    (
        "Quiero cancelar mi tarjeta de crédito",
        "Deseo dar de baja mi tarjeta de crédito",
        "Esperado: DUPLICADO"
    ),
    (
        "Necesito un préstamo personal",
        "Solicito un crédito de consumo",
        "Esperado: DUPLICADO"
    ),
    (
        "Quiero cancelar mi tarjeta",
        "Necesito un préstamo hipotecario",
        "Esperado: NO DUPLICADO"
    ),
    (
        "Reclamo por cobro indebido en mi cuenta",
        "Presento queja por cargo no reconocido",
        "Esperado: DUPLICADO"
    ),
    (
        "Actualizar mi número de teléfono",
        "Solicito información de créditos",
        "Esperado: NO DUPLICADO"
    ),
]

for t1, t2, esperado in test_cases:
    prob = predecir_duplicado(t1, t2, best_model, word_to_idx, MAX_SEQ_LEN)
    resultado = "DUPLICADO" if prob >= 0.5 else "NO DUPLICADO"
    emoji = "✓" if esperado.split(": ")[1] == resultado else "✗"
    print(f"\n{emoji} Prob: {prob:.3f} → {resultado} ({esperado})")
    print(f"  T1: {t1}")
    print(f"  T2: {t2}")
```

### Salida Esperada

```
==============================================================
MÉTRICAS DE EVALUACIÓN - DETECCIÓN DE DUPLICADOS BANCARIOS
==============================================================

  Accuracy:  0.8125  (81.2%)
  Precision: 0.8200  (82.0%)
  Recall:    0.8000  (80.0%)
  F1-Score:  0.8099  (81.0%)
  ROC-AUC:   0.8750  (87.5%)
```

Nota: Los valores exactos variarán según la ejecución, pero se espera un accuracy entre 0.70-0.92 dado el tamaño del corpus.

### Verificación

```python
assert accuracy > 0.60, f"Accuracy ({accuracy:.3f}) debe superar 0.60"
assert roc_auc > 0.60, f"ROC-AUC ({roc_auc:.3f}) debe superar 0.60"
print(f"\n{'='*60}")
print("✓ EVALUACIÓN COMPLETADA EXITOSAMENTE")
print(f"{'='*60}")
print(f"\nResumen final del sistema:")
print(f"  - Dataset: {df.shape[0]} solicitudes en {df['grupo_duplicado'].nunique()} grupos")
print(f"  - Vocabulario: {VOCAB_SIZE} tokens")
print(f"  - Embeddings: Word2Vec {EMBEDDING_DIM}D")
print(f"  - Modelo: Siamesa LSTM (64 unidades)")
print(f"  - Pares de entrenamiento: {len(y_train)}")
print(f"  - Pares de evaluación: {len(y_test)}")
print(f"  - Accuracy: {accuracy:.4f}")
print(f"  - F1-Score: {f1:.4f}")
print(f"  - ROC-AUC: {roc_auc:.4f}")
```

---

## Paso 9: Manejo de Tokens Fuera de Vocabulario (OOV)

### Objetivo

Implementar una estrategia robusta para manejar tokens que no aparecen en el vocabulario de entrenamiento (Out-Of-Vocabulary), asegurando que el sistema funcione correctamente con solicitudes que contengan palabras nuevas.

### Instrucciones

1. Analiza el impacto de tokens fuera de vocabulario en el sistema:

```python
# Demostrar el manejo de tokens OOV
print("=" * 60)
print("MANEJO DE TOKENS FUERA DE VOCABULARIO (OOV)")
print("=" * 60)

# Ejemplos con palabras que probablemente no están en el vocabulario
textos_con_oov = [
    "Solicito refinanciamiento de mi deuda hipotecaria vigente",
    "Necesito una reestructuración de mis obligaciones crediticias",
    "Quiero tramitar un microcrédito para emprendimiento",
    "Requiero información sobre criptomonedas y blockchain",
]

print("\nAnálisis de tokens OOV por texto:")
for texto in textos_con_oov:
    tokens = preprocesar_texto(texto)
    oov_tokens = [t for t in tokens if t not in word_to_idx or word_to_idx[t] == word_to_idx["<UNK>"]]
    in_vocab = [t for t in tokens if t in word_to_idx and word_to_idx[t] != word_to_idx["<UNK>"]]
    
    print(f"\n  Texto: '{texto}'")
    print(f"  Tokens totales: {len(tokens)}")
    print(f"  En vocabulario: {in_vocab}")
    print(f"  Fuera de vocabulario (OOV): {oov_tokens}")
    print(f"  Tasa OOV: {len(oov_tokens)/max(len(tokens),1)*100:.1f}%")
```

2. Implementa estrategias mejoradas para tokens OOV:

```python
def texto_a_secuencia_robusto(texto, word_to_idx, max_len, w2v_model=None):
    """
    Convierte texto a secuencia con manejo robusto de tokens OOV.
    
    Estrategias para tokens OOV:
    1. Buscar subpalabras que estén en el vocabulario
    2. Asignar índice UNK si no se encuentra coincidencia
    
    Args:
        texto: texto preprocesado (string de tokens separados por espacio)
        word_to_idx: diccionario palabra -> índice
        max_len: longitud máxima de secuencia
        w2v_model: modelo Word2Vec (opcional, para búsqueda de similares)
    
    Returns:
        secuencia: lista de índices numéricos
        oov_count: número de tokens OOV encontrados
    """
    tokens = texto.split() if isinstance(texto, str) else texto
    secuencia = []
    oov_count = 0
    
    for token in tokens:
        if token in word_to_idx:
            secuencia.append(word_to_idx[token])
        else:
            # Estrategia 1: buscar subpalabra más larga en vocabulario
            found = False
            if len(token) > 3:
                for sub_len in range(len(token) - 1, 2, -1):
                    sub_token = token[:sub_len]
                    if sub_token in word_to_idx:
                        secuencia.append(word_to_idx[sub_token])
                        found = True
                        break
            
            # Estrategia 2: buscar palabra más similar en Word2Vec
            if not found and w2v_model is not None:
                try:
                    similares = w2v_model.wv.most_similar(token, topn=1)
                    palabra_similar = similares[0][0]
                    if palabra_similar in word_to_idx:
                        secuencia.append(word_to_idx[palabra_similar])
                        found = True
                except (KeyError, IndexError):
                    pass
            
            # Fallback: usar token UNK
            if not found:
                secuencia.append(word_to_idx["<UNK>"])
                oov_count += 1
    
    return secuencia, oov_count

# Probar con textos que contienen OOV
print("\nPrueba de conversión robusta:")
for texto in textos_con_oov:
    tokens = preprocesar_texto(texto)
    texto_limpio = " ".join(tokens)
    seq, oov = texto_a_secuencia_robusto(texto_limpio, word_to_idx, MAX_SEQ_LEN, w2v_model)
    print(f"\n  '{texto[:50]}...'")
    print(f"  Secuencia: {seq[:8]}... | OOV restantes: {oov}")
```

3. Evalúa el impacto de tokens OOV en las predicciones:

```python
# Predicción con textos que contienen OOV
print("\nPredicciones con textos que contienen tokens OOV:")
print("-" * 60)

oov_test_cases = [
    (
        "Solicito refinanciamiento de mi deuda",
        "Necesito reestructurar mi préstamo actual",
        "Esperado: DUPLICADO (ambos sobre reestructuración de deuda)"
    ),
    (
        "Quiero tramitar un microcrédito",
        "Deseo cancelar mi cuenta corriente",
        "Esperado: NO DUPLICADO (diferentes categorías)"
    ),
]

for t1, t2, esperado in oov_test_cases:
    prob = predecir_duplicado(t1, t2, best_model, word_to_idx, MAX_SEQ_LEN)
    resultado = "DUPLICADO" if prob >= 0.5 else "NO DUPLICADO"
    print(f"\n  Prob: {prob:.3f} → {resultado}")
    print(f"  T1: {t1}")
    print(f"  T2: {t2}")
    print(f"  {esperado}")

print("\n" + "=" * 60)
print("NOTA SOBRE TOKENS OOV EN PRODUCCIÓN:")
print("=" * 60)
print("""
En un sistema de producción, las estrategias para mitigar el impacto de
tokens OOV incluyen:

1. Ampliar el corpus de entrenamiento con más solicitudes bancarias
2. Usar embeddings pre-entrenados más grandes (FastText para español)
3. Implementar subword tokenization (BPE, WordPiece)
4. Reentrenar periódicamente el modelo con datos nuevos
5. Usar el índice UNK con un embedding promedio del vocabulario

El token <UNK> (índice 1) en nuestra implementación captura todas las
palabras desconocidas, asignándoles un vector que se ajusta durante el
entrenamiento para representar "palabra desconocida en contexto bancario".
""")
```

### Salida Esperada

```
==============================================================
MANEJO DE TOKENS FUERA DE VOCABULARIO (OOV)
==============================================================

Análisis de tokens OOV por texto:

  Texto: 'Solicito refinanciamiento de mi deuda hipotecaria vigente'
  Tokens totales: 5
  En vocabulario: ['solicito', 'mi']
  Fuera de vocabulario (OOV): ['refinanciamiento', 'deuda', 'vigente']
  Tasa OOV: 60.0%
```

### Verificación

```python
# Verificar que el sistema maneja OOV sin errores
for texto in textos_con_oov:
    tokens = preprocesar_texto(texto)
    texto_limpio = " ".join(tokens)
    seq, oov = texto_a_secuencia_robusto(texto_limpio, word_to_idx, MAX_SEQ_LEN)
    assert len(seq) > 0, "La secuencia no debe estar vacía"
    assert all(0 <= idx < VOCAB_SIZE for idx in seq), "Todos los índices deben ser válidos"

# Verificar predicción con OOV no produce errores
prob = predecir_duplicado(
    "Solicito refinanciamiento hipotecario urgente",
    "Necesito reestructurar mi crédito de vivienda",
    best_model, word_to_idx, MAX_SEQ_LEN
)
assert 0.0 <= prob <= 1.0, "La probabilidad debe estar en [0, 1]"
print("✓ Manejo de tokens OOV implementado correctamente")
```

---

## Paso 10: Resumen Final y Conclusiones

### Objetivo

Consolidar todo el trabajo realizado, documentar los artefactos generados y establecer próximos pasos para mejorar el sistema.

### Instrucciones

1. Genera el resumen ejecutivo del laboratorio:

```python
print("=" * 70)
print("   RESUMEN EJECUTIVO - SISTEMA DE DETECCIÓN DE DUPLICADOS BANCARIOS")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ PIPELINE IMPLEMENTADO                                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. Generación de datos  →  {df.shape[0]} solicitudes, 5 categorías         │
│  2. Preprocesamiento NLP →  Tokenización + Normalización + Stopwords │
│  3. Word2Vec embeddings  →  {EMBEDDING_DIM}D, {VOCAB_SIZE} tokens, Skip-gram          │
│  4. Generación de pares  →  {len(labels)} pares balanceados (50/50)        │
│  5. Red Siamesa LSTM     →  Embedding + LSTM(64) + L1 + Dense       │
│  6. Entrenamiento        →  {len(history.history['loss'])} épocas, callbacks adaptativos     │
│  7. Evaluación           →  Acc={accuracy:.3f}, F1={f1:.3f}, AUC={roc_auc:.3f}       │
│  8. Manejo OOV           →  Token <UNK> + estrategias de fallback    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\nARTEFACTOS GENERADOS:")
print(f"  📁 {RAW_DIR}/banking_requests.csv")
print(f"  📁 {PROCESSED_DIR}/banking_requests_processed.csv")
print(f"  📁 {EMBEDDINGS_DIR}/banking_w2v.model")
print(f"  📁 {EMBEDDINGS_DIR}/embedding_matrix.npy")
print(f"  📁 {EMBEDDINGS_DIR}/word_to_idx.json")
print(f"  📁 {MODELS_DIR}/siamese_best.keras")
print(f"  📁 {MODELS_DIR}/siamese_final.keras")
print(f"  📁 {MODELS_DIR}/training_history.csv")

print(f"""
INTERPRETACIÓN EN CONTEXTO BANCARIO:
─────────────────────────────────────
• Precision ({precision:.3f}): De las solicitudes marcadas como duplicadas,
  el {precision*100:.1f}% realmente lo son. Importante para no fusionar
  solicitudes diferentes por error.

• Recall ({recall:.3f}): De todas las solicitudes duplicadas reales,
  el sistema detecta el {recall*100:.1f}%. Importante para reducir la
  carga operativa de procesar solicitudes repetidas.

• F1-Score ({f1:.3f}): Balance entre precision y recall, útil como
  métrica única de rendimiento del sistema.

• ROC-AUC ({roc_auc:.3f}): Capacidad discriminativa del modelo
  independiente del umbral de decisión.

LIMITACIONES Y PRÓXIMOS PASOS:
──────────────────────────────
1. Corpus pequeño → Ampliar con datos reales (anonimizados)
2. Word2Vec básico → Migrar a FastText o transformers (BETO/RoBERTa)
3. LSTM simple → Explorar Bi-LSTM o mecanismos de atención
4. Umbral fijo (0.5) → Optimizar umbral según costo de negocio
5. Sin validación cruzada → Implementar k-fold para estimación robusta
""")
```

2. Verificación final del sistema completo:

```python
# Verificación integral
print("\n" + "=" * 60)
print("VERIFICACIÓN INTEGRAL DEL SISTEMA")
print("=" * 60)

checks = [
    ("Dataset generado", df.shape[0] > 150),
    ("Preprocesamiento aplicado", "tokens" in df.columns),
    ("Vocabulario construido", VOCAB_SIZE > 50),
    ("Embeddings entrenados", embedding_matrix.shape[1] == EMBEDDING_DIM),
    ("Pares generados", len(labels) > 100),
    ("Modelo entrenado", len(history.history['loss']) > 5),
    ("Accuracy > 0.60", accuracy > 0.60),
    ("ROC-AUC > 0.60", roc_auc > 0.60),
    ("Modelo guardado", os.path.exists(os.path.join(MODELS_DIR, "siamese_best.keras"))),
    ("Predicción funcional", 0 <= prob <= 1),
]

all_passed = True
for check_name, check_result in checks:
    status = "✓" if check_result else "✗"
    if not check_result:
        all_passed = False
    print(f"  {status} {check_name}")

print(f"\n{'='*60}")
if all_passed:
    print("  ✓ TODAS LAS VERIFICACIONES PASARON EXITOSAMENTE")
    print("  ✓ LABORATORIO COMPLETADO")
else:
    print("  ⚠ ALGUNAS VERIFICACIONES FALLARON - REVISAR PASOS ANTERIORES")
print(f"{'='*60}")
```

### Salida Esperada

Todas las verificaciones deben mostrar ✓ y el mensaje final debe indicar que el laboratorio se completó exitosamente.

---

## Resumen de Entregables

| Artefacto | Ubicación | Descripción |
|-----------|-----------|-------------|
| Dataset crudo | `data/raw/banking_requests.csv` | 185+ solicitudes bancarias sintéticas |
| Dataset procesado | `data/processed/banking_requests_processed.csv` | Textos tokenizados y normalizados |
| Modelo Word2Vec | `embeddings/banking_w2v.model` | Embeddings 100D entrenados en corpus bancario |
| Matriz embeddings | `embeddings/embedding_matrix.npy` | Matriz NumPy para capa Embedding de Keras |
| Vocabulario | `embeddings/word_to_idx.json` | Mapeo palabra → índice |
| Modelo siamés (mejor) | `models/siamese_best.keras` | Mejor checkpoint por val_accuracy |
| Modelo siamés (final) | `models/siamese_final.keras` | Modelo al finalizar entrenamiento |
| Historial | `models/training_history.csv` | Métricas por época |
| Notebook completo | `notebooks/siamese_banking_duplicates.ipynb` | Todo el código del laboratorio |
| Gráficos | `notebooks/*.png` | Visualizaciones de EDA y evaluación |

## Criterios de Evaluación

| Criterio | Peso | Descripción |
|----------|------|-------------|
| Pipeline de preprocesamiento | 20% | Correcto y completo para español bancario |
| Embeddings Word2Vec | 15% | Entrenamiento correcto y matriz compatible |
| Generación de pares | 15% | Balance de clases y correctitud de etiquetas |
| Arquitectura siamesa | 20% | Diseño correcto con LSTM compartido |
| Entrenamiento con callbacks | 15% | Convergencia y prevención de sobreajuste |
| Evaluación e interpretación | 15% | Métricas correctas y análisis contextual |

---
